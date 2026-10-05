#!/usr/bin/env python3

"""Serve batched GPU inner-product search over monthly FAISS indexes."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import faiss
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="[search-gpu] %(asctime)s %(message)s")
logger = logging.getLogger(__name__)

_DTYPE = {"fp32": torch.float32, "fp16": torch.float16}[os.environ.get("SEARCH_DTYPE", "fp32")]

_MAX_BATCH = int(os.environ.get("SEARCH_MAX_BATCH", "64"))

_BATCH_WINDOW_S = float(os.environ.get("SEARCH_BATCH_WINDOW_MS", "5")) / 1000.0

_REQUEST_TIMEOUT_S = float(os.environ.get("SEARCH_REQUEST_TIMEOUT_S", "120"))


class _Shard:
    """One month's resident vector matrix on a specific GPU."""

    def __init__(self, month: str, tensor: torch.Tensor, device: str):
        self.month = month
        self.tensor = tensor
        self.device = device
        self.n_rows = tensor.shape[0]


@dataclass
class _Pending:
    """One in-flight /search request awaiting its batch."""

    query: torch.Tensor
    top_k: int

    specs: list[tuple[str, int, int]]
    future: asyncio.Future = field(repr=False)

    slots: list[tuple[tuple[str, int, int], int]] = field(default_factory=list)


class SearchState:
    def __init__(self, index_root: Path, devices: list[str]):
        self.shards: dict[str, _Shard] = {}
        self.dim: int | None = None
        self.devices = devices
        self._load(index_root)

        self._scores_buf: dict[str, torch.Tensor] = {}
        self._query_buf: dict[str, torch.Tensor] = {}
        for dev in self.devices:
            max_rows = max((s.n_rows for s in self.shards.values() if s.device == dev), default=0)
            if max_rows == 0:
                continue
            self._scores_buf[dev] = torch.empty(
                (max_rows, _MAX_BATCH), dtype=_DTYPE, device=dev)
            self._query_buf[dev] = torch.zeros(
                (_MAX_BATCH, self.dim), dtype=_DTYPE, device=dev)
            logger.info("scratch %s: scores (%d, %d) %.0f MB + queries (%d, %d) %.1f MB",
                        dev, max_rows, _MAX_BATCH,
                        self._scores_buf[dev].element_size() * self._scores_buf[dev].nelement() / 1e6,
                        _MAX_BATCH, self.dim,
                        self._query_buf[dev].element_size() * self._query_buf[dev].nelement() / 1e6)

        self.stat_requests = 0
        self.stat_batches = 0
        self.stat_batched_requests = 0
        self._warmup()

    def _load(self, index_root: Path) -> None:
        month_dirs = sorted(
            d for d in index_root.iterdir()
            if d.is_dir() and (d / "faiss.index").exists()
        )
        if not month_dirs:
            raise RuntimeError(f"no month indexes under {index_root}")
        faiss.omp_set_num_threads(min(32, os.cpu_count() or 8))
        t0 = time.time()
        for i, d in enumerate(month_dirs):
            month = d.name

            device = self.devices[i % len(self.devices)]
            idx = faiss.read_index(str(d / "faiss.index"),
                                   faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY)
            n, dim = idx.ntotal, idx.d
            self.dim = dim if self.dim is None else self.dim
            if dim != self.dim:
                raise RuntimeError(f"{month}: dim {dim} != {self.dim}")
            vecs = idx.reconstruct_n(0, n)
            del idx
            tensor = torch.from_numpy(vecs).to(device=device, dtype=_DTYPE)
            self.shards[month] = _Shard(month, tensor, device)
            logger.info("loaded %s: %d rows -> %s (%.1f GB, %.0fs elapsed)",
                        month, n, device,
                        tensor.element_size() * tensor.nelement() / 1e9, time.time() - t0)
        total = sum(s.n_rows for s in self.shards.values())
        logger.info("READY: %d months, %d vectors, dim=%d, dtype=%s, devices=%s (%.0fs)",
                    len(self.shards), total, self.dim, _DTYPE, self.devices, time.time() - t0)

    @torch.inference_mode()
    def _warmup(self) -> None:
        """Warm up each GPU with a full batch on its largest shard."""
        for dev in self.devices:
            if dev not in self._scores_buf:
                continue
            biggest = max((s for s in self.shards.values() if s.device == dev),
                          key=lambda s: s.n_rows)
            qb, sb = self._query_buf[dev], self._scores_buf[dev]
            qb.normal_()
            torch.matmul(biggest.tensor, qb.t(), out=sb[: biggest.n_rows])
            topv, topi = torch.topk(sb[: biggest.n_rows], min(64, biggest.n_rows), dim=0)
            _ = topv.cpu(), topi.cpu()
            torch.cuda.synchronize(dev)
            logger.info("warmup %s: full batch=%d on %s (%d rows) OK; peak GPU mem %.1f GB",
                        dev, _MAX_BATCH, biggest.month, biggest.n_rows,
                        torch.cuda.max_memory_allocated(dev) / 1e9)

    @torch.inference_mode()
    def search_batch(self, batch: list[_Pending]) -> list[list]:
        """Return each request's search hits in batch order."""

        groups: dict[tuple[str, int, int], list[_Pending]] = {}
        for req in batch:
            req.slots = []
            for month, lo, hi in req.specs:
                shard = self.shards.get(month)
                if shard is None:
                    continue
                lo = max(0, lo)
                hi = min(shard.n_rows, hi)
                if hi <= lo:
                    continue
                key = (month, lo, hi)
                members = groups.setdefault(key, [])
                members.append(req)
                req.slots.append((key, len(members) - 1))

        results: dict[tuple[str, int, int], tuple[torch.Tensor, torch.Tensor, int]] = {}
        for key, members in groups.items():
            month, lo, hi = key
            shard = self.shards[month]
            rows = hi - lo
            b = len(members)
            k = min(max(m.top_k for m in members), rows)
            if k <= 0:
                continue
            qb = self._query_buf[shard.device]
            sb = self._scores_buf[shard.device]

            qb[:b].copy_(torch.stack([m.query for m in members]).to(dtype=_DTYPE))

            torch.matmul(shard.tensor[lo:hi], qb.t(), out=sb[:rows])

            topv, topi = torch.topk(sb[:rows, :b], k, dim=0)
            results[key] = (topv, topi + lo, k)

        cpu_results: dict[tuple[str, int, int], tuple[list, list, int]] = {}
        for key, (topv, topi, k) in results.items():
            cpu_results[key] = (topv.float().cpu().tolist(), topi.cpu().tolist(), k)

        out: list[list] = []
        for req in batch:
            merged_scores: list[float] = []
            merged_meta: list[tuple[str, int]] = []
            for key, slot in req.slots:
                got = cpu_results.get(key)
                if got is None:
                    continue
                topv_l, topi_l, k = got
                month, lo, hi = key
                k_req = min(req.top_k, hi - lo, k)
                for i in range(k_req):
                    merged_scores.append(topv_l[i][slot])
                    merged_meta.append((month, topi_l[i][slot]))
            if not merged_scores:
                out.append([])
                continue
            order = np.argsort(np.asarray(merged_scores))[::-1][: req.top_k]
            out.append([[merged_scores[j], merged_meta[j][0], merged_meta[j][1]]
                        for j in order])
        return out


class SearchRequest(BaseModel):
    query: list[float]
    shards: list[dict]
    top_k: int = 10


def build_app(state: SearchState) -> FastAPI:
    queue: asyncio.Queue[_Pending] = asyncio.Queue()

    gpu_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="search-gpu")

    async def _batcher() -> None:
        """Batch queued requests and dispatch them to the GPU worker."""
        loop = asyncio.get_running_loop()
        while True:
            first = await queue.get()
            if _BATCH_WINDOW_S > 0:
                await asyncio.sleep(_BATCH_WINDOW_S)
            batch = [first]
            while len(batch) < _MAX_BATCH:
                try:
                    batch.append(queue.get_nowait())
                except asyncio.QueueEmpty:
                    break
            state.stat_batches += 1
            state.stat_requests += len(batch)
            if len(batch) > 1:
                state.stat_batched_requests += len(batch)
            try:
                hits = await loop.run_in_executor(gpu_pool, state.search_batch, batch)
                for req, h in zip(batch, hits):
                    if not req.future.done():
                        req.future.set_result(h)
            except Exception as e:  # noqa: BLE001 -- keep serving after batch errors
                logger.exception("batch of %d failed", len(batch))
                for req in batch:
                    if not req.future.done():
                        req.future.set_exception(e)

    @contextlib.asynccontextmanager
    async def lifespan(_app: FastAPI):
        task = asyncio.create_task(_batcher())
        yield
        task.cancel()
        gpu_pool.shutdown(wait=False)

    app = FastAPI(lifespan=lifespan)

    @app.get("/health")
    def health() -> dict:
        served = max(1, state.stat_requests)
        return {
            "status": "ok",
            "months": sorted(state.shards),
            "n_vectors": sum(s.n_rows for s in state.shards.values()),
            "dim": state.dim,
            "dtype": str(_DTYPE),
            "batching": {
                "max_batch": _MAX_BATCH,
                "window_ms": _BATCH_WINDOW_S * 1000,
                "requests": state.stat_requests,
                "batches": state.stat_batches,
                "avg_batch": round(state.stat_requests / max(1, state.stat_batches), 2),
                "batched_frac": round(state.stat_batched_requests / served, 3),
            },
        }

    @app.post("/search")
    async def search(req: SearchRequest) -> dict:
        q = np.asarray(req.query, dtype=np.float32)
        if state.dim is not None and q.shape[0] != state.dim:
            return {"hits": [], "error": f"query dim {q.shape[0]} != {state.dim}"}
        pending = _Pending(
            query=torch.from_numpy(q),
            top_k=req.top_k,
            specs=[(str(s["month"]), int(s.get("lo", 0)), int(s.get("hi", 1 << 62)))
                   for s in req.shards if "month" in s],
            future=asyncio.get_running_loop().create_future(),
        )
        await queue.put(pending)
        try:
            hits = await asyncio.wait_for(pending.future, timeout=_REQUEST_TIMEOUT_S)
        except TimeoutError:
            return {"hits": [], "error": f"search timed out after {_REQUEST_TIMEOUT_S}s"}
        return {"hits": hits}

    return app


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default=os.environ.get("SEARCH_HOST", "0.0.0.0"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("SEARCH_PORT", "8100")))
    ap.add_argument("--index-root", default=os.environ.get(
        "SEARCH_INDEX_ROOT", "/datasets/faiss_index"))
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA device visible to the search service")
    devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    logger.info("devices visible: %s (CUDA_VISIBLE_DEVICES=%s)",
                devices, os.environ.get("CUDA_VISIBLE_DEVICES", "(unset)"))
    logger.info("batching: max_batch=%d window=%.1fms", _MAX_BATCH, _BATCH_WINDOW_S * 1000)
    state = SearchState(Path(args.index_root), devices)
    app = build_app(state)
    logger.info("serving on %s:%d", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")

if __name__ == "__main__":
    main()
