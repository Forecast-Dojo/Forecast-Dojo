"""Retrieve dated evidence with dense and lexical search."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import bm25s
import pyarrow.parquet as pq
import tiktoken
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from environment.tool import RetrieverToolkit

logger = logging.getLogger(__name__)

DEFAULT_K_PER_QUERY = 10
DEFAULT_N_FINAL = 15
DEFAULT_CORPUS_START_DATE = "2025-03-01"
DEFAULT_RRF_K = 60
DEFAULT_TEXT_TOKEN_BUDGET = 1500
DEFAULT_EMBEDDING_ENDPOINT = "http://localhost:8001/v1"
DEFAULT_EMBEDDING_MODEL = "Qwen3-Embedding-8B"
DEFAULT_INDEX_ROOT = Path("indexes/faiss")
DEFAULT_BM25_ROOT = Path("indexes/bm25")

TOKEN_ENCODING = tiktoken.encoding_for_model("gpt-4")


@dataclass
class _BM25Shard:
    """Memory-mapped BM25 index with row-aligned publication dates."""

    name: str
    bm25: bm25s.BM25
    published_dates: list[str]  # Aligned to BM25 row IDs.

    def __len__(self) -> int:
        return len(self.published_dates)


class _BM25Corpus:
    """Lazily load and cache dated BM25 shards under a shared load lock."""

    def __init__(
        self,
        bm25_root: Path,
        index_root: Path,
    ):
        self.bm25_root = Path(bm25_root)
        self.index_root = Path(index_root)
        self._shards: dict[str, _BM25Shard] = {}
        # Serialize cold loads to avoid duplicate shard allocations.
        self._load_lock = threading.Lock()

    def list_shards(self) -> list[str]:
        if not self.bm25_root.is_dir():
            return []
        return sorted(
            d.name for d in self.bm25_root.iterdir()
            if d.is_dir() and not d.name.startswith("_")
        )

    def get(self, name: str) -> _BM25Shard | None:
        cached = self._shards.get(name)
        if cached is not None:
            return cached
        # Recheck after acquiring the lock.
        with self._load_lock:
            cached = self._shards.get(name)
            if cached is not None:
                return cached
            bm25_dir = self.bm25_root / name / "bm25s"
            parquet = self.index_root / name / "texts.parquet"
            if not bm25_dir.exists() or not parquet.exists():
                return None
            try:
                bm25 = bm25s.BM25.load(str(bm25_dir), mmap=True)
            except Exception as e:
                logger.warning("[bm25] failed to load %s: %s", name, e)
                return None
            # Read dates from the same parquet rows used to build the lexical index.
            try:
                available = set(pq.read_schema(parquet).names)
                if "published_date" in available:
                    pub = pq.read_table(parquet, columns=["published_date"]) \
                        .column("published_date").to_pylist()
                else:
                    pub = [""] * bm25.scores["data"].shape[0]
            except Exception as e:
                logger.warning("[bm25] failed to read published_date for %s: %s", name, e)
                return None
            shard = _BM25Shard(name=name, bm25=bm25, published_dates=pub)
            self._shards[name] = shard
            logger.info("[bm25] loaded shard %s (%d docs)", name, len(shard))
            return shard


@dataclass
class _HitKey:
    shard: str
    row: int

    def __hash__(self) -> int:
        return hash((self.shard, self.row))


@dataclass
class _HitAccumulator:
    """Accumulate RRF scores and query coverage for one document."""

    rrf_sum: float = 0.0
    n_query_hits: int = 0
    best_dense: float | None = None
    best_bm25: float | None = None
    queries_seen: set[int] = field(default_factory=set)


class HybridRetriever:
    """Combine dense and BM25 rankings within each question's date window."""

    def __init__(
        self,
        *,
        index_root: Path,
        bm25_root: Path,
        embedding_endpoint: str,
        embedding_model: str,
        k_per_query: int,
        n_final: int,
        rrf_k: int,
        text_token_budget: int,
        dense_concurrency: int,
        timeout: int = 120,
    ):
        self._dense = RetrieverToolkit(
            embedding_endpoint=embedding_endpoint,
            index_root=index_root,
            embedding_model=embedding_model,
            text_token_budget=text_token_budget,
            timeout=timeout,
            concurrency=dense_concurrency,
            # Evidence screening uses unmodified similarity rankings.
            decay_alpha=1.0,
        )
        self._bm25 = _BM25Corpus(bm25_root=bm25_root, index_root=index_root)

        self.k_per_query = int(k_per_query)
        self.n_final = int(n_final)
        self.rrf_k = int(rrf_k)
        self.text_token_budget = int(text_token_budget)

    async def prewarm_dense(self, dates: list[str] | None = None) -> int:
        return await self._dense.prewarm(dates=dates)

    async def cleanup(self) -> None:
        await self._dense.cleanup()


    async def _dense_search(
        self, query: str, *, start: str, cutoff: str,
    ) -> list[tuple[float, str, int]]:
        """Return (similarity, shard, row) hits using task-local date limits."""
        # Task-local windows keep concurrent queries isolated.
        from environment.tool import (
            _window_start, _window_cutoff,
        )
        tok_s = _window_start.set(start)
        tok_c = _window_cutoff.set(cutoff)
        try:
            return await self._dense._search_all(query, self.k_per_query)
        finally:
            _window_start.reset(tok_s)
            _window_cutoff.reset(tok_c)


    def _bm25_search_one_shard(
        self,
        shard: _BM25Shard,
        tokens: bm25s.tokenization.Tokenized,
        *,
        start: str,
        cutoff: str,
    ) -> list[tuple[float, str, int]]:
        """Retrieve an oversized BM25 candidate list, then filter by publication date."""
        oversample = max(self.k_per_query * 5, 50)
        try:
            results, scores = shard.bm25.retrieve(
                tokens, k=oversample, show_progress=False,
            )
        except ValueError:
            return []
        rows = results[0].tolist()
        sims = scores[0].tolist()
        out: list[tuple[float, str, int]] = []
        for row, score in zip(rows, sims):
            if row < 0 or row >= len(shard):
                continue
            pd_str = shard.published_dates[row]
            if not pd_str:
                continue
            if pd_str < start or pd_str > cutoff:
                continue
            out.append((float(score), shard.name, int(row)))
            if len(out) >= self.k_per_query:
                break
        return out

    def _bm25_search(
        self, query: str, *, start: str, cutoff: str,
    ) -> list[tuple[float, str, int]]:
        """Merge dated BM25 candidates across shards and return the top K."""
        shard_names = [
            n for n in self._bm25.list_shards()
            if (start[:7] <= n[:7] <= cutoff[:7])
        ]
        if not shard_names:
            return []
        tokens = bm25s.tokenize(
            query, stopwords="en", lower=True, show_progress=False,
        )
        flat: list[tuple[float, str, int]] = []
        for name in shard_names:
            shard = self._bm25.get(name)
            if shard is None:
                continue
            flat.extend(self._bm25_search_one_shard(
                shard, tokens, start=start, cutoff=cutoff,
            ))
        flat.sort(key=lambda x: x[0], reverse=True)
        return flat[: self.k_per_query]


    @staticmethod
    def _rrf_increment(
        acc: dict[_HitKey, _HitAccumulator],
        ranked: list[tuple[float, str, int]],
        *,
        rrf_k: int,
        query_idx: int,
        kind: str,
    ) -> None:
        """Add reciprocal-rank contributions from one query/retriever ranking."""
        for rank, (score, shard, row) in enumerate(ranked, start=1):
            key = _HitKey(shard=shard, row=row)
            entry = acc.get(key)
            if entry is None:
                entry = _HitAccumulator()
                acc[key] = entry
            entry.rrf_sum += 1.0 / (rrf_k + rank)
            if query_idx not in entry.queries_seen:
                entry.queries_seen.add(query_idx)
                entry.n_query_hits += 1
            if kind == "dense":
                if entry.best_dense is None or score > entry.best_dense:
                    entry.best_dense = float(score)
            else:
                if entry.best_bm25 is None or score > entry.best_bm25:
                    entry.best_bm25 = float(score)


    async def retrieve_for_question(
        self,
        *,
        event_id: str,
        slug: str,
        title: str,
        queries: list[str],
        start_date: str,
        cutoff_date: str,
    ) -> dict[str, Any]:
        """Retrieve and fuse query rankings into one event evidence record."""
        if not queries:
            return _empty_record(event_id, slug, title, start_date, cutoff_date,
                                 reason="no_queries")

        acc: dict[_HitKey, _HitAccumulator] = {}
        n_dense_hits = 0
        n_bm25_hits = 0
        n_zero_hit_queries = 0

        dense_tasks = [
            asyncio.create_task(
                self._dense_search(q, start=start_date, cutoff=cutoff_date)
            )
            for q in queries
        ]
        # Run lexical search in threads to keep the event loop responsive.
        bm25_tasks = [
            asyncio.create_task(
                asyncio.to_thread(
                    self._bm25_search, q,
                    start=start_date, cutoff=cutoff_date,
                )
            )
            for q in queries
        ]
        dense_per_query = await asyncio.gather(
            *dense_tasks, return_exceptions=False,
        )
        bm25_per_query = await asyncio.gather(
            *bm25_tasks, return_exceptions=False,
        )

        for qi, q in enumerate(queries):
            d_hits = dense_per_query[qi]
            b_hits = bm25_per_query[qi]
            n_dense_hits += len(d_hits)
            n_bm25_hits += len(b_hits)
            if not d_hits and not b_hits:
                n_zero_hit_queries += 1
                logger.debug("query %d/%d returned 0 hits: %r", qi, len(queries), q)
                continue
            self._rrf_increment(acc, d_hits, rrf_k=self.rrf_k,
                                query_idx=qi, kind="dense")
            self._rrf_increment(acc, b_hits, rrf_k=self.rrf_k,
                                query_idx=qi, kind="bm25")

        if not acc:
            return _empty_record(event_id, slug, title, start_date, cutoff_date,
                                 reason="all_queries_empty",
                                 stats={
                                     "queries_total": len(queries),
                                     "queries_zero_hits": n_zero_hit_queries,
                                     "dense_hits": n_dense_hits,
                                     "bm25_hits": n_bm25_hits,
                                     "fused_unique_docs": 0,
                                 })

        ranked = sorted(
            acc.items(),
            key=lambda kv: kv[1].rrf_sum,
            reverse=True,
        )[: self.n_final]

        hits: list[dict[str, Any]] = []
        for key, entry in ranked:
            shard = self._dense._shards.get(key.shard)
            if shard is None:
                # BM25-only hits still need their article metadata loaded.
                shard = self._dense._load_shard(key.shard)
                if shard is None:
                    logger.warning("could not load shard %s for hit", key.shard)
                    continue
            hits.append({
                "shard": key.shard,
                "row": key.row,
                "article_id": shard.article_ids[key.row] or None,
                "url": shard.urls[key.row] or None,
                "title": shard.titles[key.row] or None,
                "published_date": shard.published_dates[key.row] or None,
                "rrf_score": round(entry.rrf_sum, 6),
                "n_query_hits": entry.n_query_hits,
                "best_dense": (round(entry.best_dense, 6)
                               if entry.best_dense is not None else None),
                "best_bm25": (round(entry.best_bm25, 6)
                              if entry.best_bm25 is not None else None),
                "text": _truncate(shard.text_at(key.row), self.text_token_budget),
            })

        return {
            "event_id": event_id,
            "slug": slug,
            "title": title,
            "window": [start_date, cutoff_date],
            "n_queries": len(queries),
            "hits": hits,
            "stats": {
                "queries_total": len(queries),
                "queries_zero_hits": n_zero_hit_queries,
                "dense_hits": n_dense_hits,
                "bm25_hits": n_bm25_hits,
                "fused_unique_docs": len(acc),
            },
            "error": None,
        }


def _truncate(text: str | None, budget: int) -> str:
    """Truncate article text to the token budget."""
    if not text:
        return ""
    tokens = TOKEN_ENCODING.encode(text, allowed_special="all")
    if len(tokens) > budget:
        return TOKEN_ENCODING.decode(tokens[:budget]) + "..."
    return text


def _empty_record(
    event_id: str, slug: str, title: str,
    start: str, cutoff: str,
    *,
    reason: str,
    stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a zero-hit record with an error reason."""
    return {
        "event_id": event_id,
        "slug": slug,
        "title": title,
        "window": [start, cutoff],
        "n_queries": 0,
        "hits": [],
        "stats": stats or {
            "queries_total": 0,
            "queries_zero_hits": 0,
            "dense_hits": 0,
            "bm25_hits": 0,
            "fused_unique_docs": 0,
        },
        "error": reason,
    }


def _resolution_date(q: dict) -> str | None:
    """Prefer recorded event/market close dates, falling back to deadline fields."""
    cd = q.get("close_date") or ""
    if cd:
        return cd[:10]
    closes: list[str] = []
    for m in (q.get("markets") or []):
        ct = m.get("closed_time") or ""
        if ct:
            closes.append(ct[:10])
    if closes:
        return max(closes)
    for fallback in (q.get("resolution_date"), q.get("end_date")):
        if fallback:
            return fallback[:10]
    return None


def _start_date(q: dict) -> str | None:
    """Return YYYY-MM-DD of question start (when belief tracking began)."""
    raw = (q.get("start_date") or q.get("creation_date") or
           q.get("probability_start_date") or "")
    return raw[:10] if raw else None


def _shift_date(iso: str, *, days: int) -> str:
    """Shift a YYYY-MM-DD date by the requested number of days."""
    from datetime import date as _date, timedelta as _td
    y, m, d = iso[:10].split("-")
    base = _date(int(y), int(m), int(d))
    return (base + _td(days=days)).isoformat()


def _load_questions(path: Path) -> dict[str, dict]:
    """Load filtered_signal.jsonl (or similar) as event_id -> record."""
    out: dict[str, dict] = {}
    with path.open() as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            r = json.loads(ln)
            eid = str(r.get("event_id") or "")
            if eid:
                out[eid] = r
    return out


def _iter_decompositions(path: Path):
    """Yield non-error decomposition rows from the JSONL produced by decompose.py."""
    with path.open() as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            r = json.loads(ln)
            if r.get("error"):
                continue
            if not r.get("flat_queries"):
                continue
            yield r


def _load_completed_event_ids(path: Path) -> set[str]:
    """Return event_ids whose retrieval row is already in the output."""
    if not path.exists():
        return set()
    out: set[str] = set()
    with path.open() as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                r = json.loads(ln)
            except json.JSONDecodeError:
                continue
            # Resume treats all recorded retrieval attempts, including errors, as complete.
            eid = r.get("event_id")
            if eid:
                out.add(str(eid))
    return out


async def _run(args) -> None:
    # Share a larger executor between dense and lexical shard searches.
    import concurrent.futures
    loop = asyncio.get_running_loop()
    n_workers = max(args.thread_pool, args.concurrency * 8)
    loop.set_default_executor(
        concurrent.futures.ThreadPoolExecutor(
            max_workers=n_workers,
            thread_name_prefix="retrieve-io",
        )
    )
    logger.info("default thread pool sized to %d workers", n_workers)

    questions = _load_questions(args.questions)
    logger.info("loaded %d questions from %s", len(questions), args.questions)

    decomps = list(_iter_decompositions(args.decompositions))
    logger.info("loaded %d non-error decompositions from %s",
                len(decomps), args.decompositions)

    if args.question_end is not None:
        decomps = decomps[args.question_start:args.question_end]
    elif args.question_start:
        decomps = decomps[args.question_start:]

    completed = _load_completed_event_ids(args.output)
    pending = [
        d for d in decomps
        if str(d.get("event_id") or "") not in completed
        and str(d.get("event_id") or "") in questions
    ]
    logger.info("resume: %d already done; %d pending (after question slicing)",
                len(completed), len(pending))
    if args.limit is not None:
        pending = pending[: args.limit]
        logger.info("--limit applied: capped pending to %d", len(pending))
    if not pending:
        logger.info("nothing to do.")
        return

    retriever = HybridRetriever(
        index_root=args.index_root,
        bm25_root=args.bm25_root,
        embedding_endpoint=args.embedding_endpoint,
        embedding_model=args.embedding_model,
        k_per_query=args.k_per_query,
        n_final=args.n_final,
        rrf_k=args.rrf_k,
        text_token_budget=args.text_token_budget,
        dense_concurrency=args.dense_concurrency,
    )

    if args.prewarm:
        logger.info("prewarming dense shards...")
        n = await retriever.prewarm_dense()
        logger.info("prewarmed %d dense shards", n)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    n_err = 0
    write_lock = asyncio.Lock()
    sem = asyncio.Semaphore(max(1, int(args.concurrency)))

    async def _process_one(d: dict) -> dict[str, Any]:
        eid = str(d["event_id"])
        q = questions[eid]
        start = args.corpus_start_date
        resolution = _resolution_date(q)
        if not start or not resolution:
            return _empty_record(
                eid, q.get("slug") or "", q.get("title") or "",
                start or "", resolution or "",
                reason="no_window",
            )
        # Match the forecast-date buffer before the recorded close.
        cutoff = _shift_date(resolution, days=-args.min_days_before_resolution)
        if cutoff < start:
            return _empty_record(
                eid, q.get("slug") or "", q.get("title") or "",
                start, cutoff,
                reason="window_collapsed_after_buffer",
            )
        async with sem:
            try:
                return await retriever.retrieve_for_question(
                    event_id=eid,
                    slug=q.get("slug") or d.get("slug") or "",
                    title=q.get("title") or d.get("title") or "",
                    queries=list(d["flat_queries"]),
                    start_date=start,
                    cutoff_date=cutoff,
                )
            except Exception as e:
                logger.exception("retrieval failed for %s: %s", eid, e)
                rec = _empty_record(
                    eid, q.get("slug") or "", q.get("title") or "",
                    start, cutoff,
                    reason=f"retrieve_error: {type(e).__name__}: {e}",
                )
                return rec

    try:
        with args.output.open("a") as out_f, \
             logging_redirect_tqdm(), \
             tqdm(
                 total=len(pending) + len(completed),
                 initial=len(completed) if args.limit is None else 0,
                 unit="q",
                 desc=f"retrieve (concurrency={args.concurrency})",
                 dynamic_ncols=True,
                 smoothing=0.05,
             ) as bar:

            async def _runner(d):
                nonlocal n_ok, n_err
                rec = await _process_one(d)
                async with write_lock:
                    out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    out_f.flush()
                    if rec.get("error"):
                        n_err += 1
                    else:
                        n_ok += 1
                    bar.set_postfix(ok=n_ok, err=n_err, refresh=False)
                    bar.update(1)

            await asyncio.gather(*(_runner(d) for d in pending))
    finally:
        await retriever.cleanup()

    logger.info("retrieval complete: ok=%d  err=%d  out=%s",
                n_ok, n_err, args.output)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--questions", type=Path,
        default=Path("questions/polymarket/filtered_signal.jsonl"),
        help="Source questions with recorded closing dates.",
    )
    ap.add_argument(
        "--decompositions", type=Path,
        default=Path("results/decomposed_opus-4-8.jsonl"),
        help="Research queries from data.evidence_check.decompose.",
    )
    ap.add_argument(
        "--output", type=Path,
        default=Path("results/retrieved.jsonl"),
        help="Output JSONL; all recorded event IDs, including errors, are skipped on resume.",
    )
    ap.add_argument(
        "--index-root", type=Path, default=DEFAULT_INDEX_ROOT,
        help="FAISS index root.",
    )
    ap.add_argument(
        "--bm25-root", type=Path, default=DEFAULT_BM25_ROOT,
        help="Offline BM25 index root (built via build_bm25.py).",
    )
    ap.add_argument(
        "--embedding-endpoint", default=DEFAULT_EMBEDDING_ENDPOINT,
        help="vLLM embedding base URL (Qwen3-8B-Embedding).",
    )
    ap.add_argument(
        "--embedding-model", default=DEFAULT_EMBEDDING_MODEL,
        help="Served embedding model name.",
    )
    ap.add_argument(
        "--k-per-query", type=int, default=DEFAULT_K_PER_QUERY,
        help="Top-K each retriever returns per query.",
    )
    ap.add_argument(
        "--n-final", type=int, default=DEFAULT_N_FINAL,
        help="Global top-N kept per question after RRF fuse.",
    )
    ap.add_argument(
        "--rrf-k", type=int, default=DEFAULT_RRF_K,
        help="RRF smoothing constant. SIGIR-2009 default = 60.",
    )
    ap.add_argument(
        "--min-days-before-resolution", type=int, default=2,
        help="Subtract this many days from recorded close for the retrieval cutoff (default: 2).",
    )
    ap.add_argument(
        "--corpus-start-date", default=DEFAULT_CORPUS_START_DATE,
        help="Inclusive corpus-wide retrieval start date (default: 2025-03-01).",
    )
    ap.add_argument(
        "--text-token-budget", type=int, default=DEFAULT_TEXT_TOKEN_BUDGET,
        help="Per-doc text truncation in the output JSON.",
    )
    ap.add_argument(
        "--concurrency", type=int, default=8,
        help="Concurrent questions in flight.",
    )
    ap.add_argument(
        "--dense-concurrency", type=int, default=10,
        help="Concurrent embedding HTTP calls (per the dense toolkit's "
             "internal semaphore).",
    )
    ap.add_argument(
        "--thread-pool", type=int, default=128,
        help="FAISS/BM25 worker threads; raised to at least concurrency * 8.",
    )
    ap.add_argument(
        "--prewarm", action="store_true",
        help="Load all dense shards before retrieval.",
    )
    ap.add_argument(
        "--question-start", type=int, default=0,
        help="0-indexed first decomposition row (post-error-filter).",
    )
    ap.add_argument(
        "--question-end", type=int, default=None,
        help="0-indexed last decomposition row, exclusive.",
    )
    ap.add_argument(
        "--limit", type=int, default=None,
        help="Maximum questions after resume filtering.",
    )
    ap.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = ap.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s [%(name)s] %(message)s",
    )
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
