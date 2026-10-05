"""Build dated FAISS indexes and aligned article sidecars."""

from __future__ import annotations

import os

# Set before importing tokenizers or transformers.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

import argparse
import json
import re
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import pandas as pd
from tqdm import tqdm


DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"

EMBED_BATCH_SIZE = 2048
MAX_TOKENS = 32000
GPU_MEMORY_UTILIZATION = float(os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", "0.95"))

# vLLM scheduling limits, independent of the submission batch size.
MAX_NUM_BATCHED_TOKENS = int(os.environ.get("VLLM_MAX_NUM_BATCHED_TOKENS", "131072"))
MAX_NUM_SEQS = int(os.environ.get("VLLM_MAX_NUM_SEQS", "512"))

RENDER_POLL_SEC = 0.5


# Match the serving tokenizer, budget, and truncation suffix.
SCRAPE_TOKENIZER = "gpt-4"          # cl100k_base
SCRAPE_TOKEN_BUDGET = 8000


def _scrape_encoding_name() -> str:
    """Return the canonical tiktoken encoding name used by serving."""
    import tiktoken

    return tiktoken.encoding_for_model(SCRAPE_TOKENIZER).name


def _precompute_truncated(texts: list[str], budget: int) -> tuple[list[str], list[int]]:
    """Return serving-compatible truncated bodies and original token lengths."""
    import tiktoken

    enc = tiktoken.encoding_for_model(SCRAPE_TOKENIZER)
    out: list[str] = []
    lengths: list[int] = []
    token_lists = enc.encode_batch(
        [t if t else "" for t in texts], allowed_special="all"
    )
    over_idx = [i for i, toks in enumerate(token_lists) if len(toks) > budget]
    decoded_over = dict(zip(
        over_idx,
        enc.decode_batch([token_lists[i][:budget] for i in over_idx]),
    ))
    for i, (t, toks) in enumerate(zip(texts, token_lists)):
        lengths.append(len(toks))
        if not t:
            out.append("")
        elif len(toks) > budget:
            out.append(decoded_over[i] + "...")
        else:
            out.append(t)
    return out, lengths


def date_label(path: Path) -> str:
    """Return the label of a feather file or directory."""
    return path.stem if path.suffix == ".feather" else path.name


_MONTH_RE = re.compile(r"^\d{4}-\d{2}$")
_DAY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def is_month_label(date_name: str) -> bool:
    """Check whether a shard label has YYYY-MM granularity."""
    return bool(_MONTH_RE.match(date_name))


def resolve_date_path(data_root: Path, name: str) -> Path | None:
    """Prefer <name>.feather, then <name>/; return None if neither exists."""
    flat = data_root / f"{name}.feather"
    if flat.is_file():
        return flat
    d = data_root / name
    if d.is_dir():
        return d
    return None


def list_date_dirs(data_root: Path) -> list[Path]:
    """List feather files and subdirectories, excluding underscore-prefixed dirs."""
    out: list[Path] = []
    for p in sorted(data_root.iterdir()):
        if p.is_file() and p.suffix == ".feather":
            out.append(p)
        elif p.is_dir() and not p.name.startswith("_"):
            out.append(p)
    return out


def index_path(index_root: Path, date_name: str) -> Path:
    return index_root / date_name / "faiss.index"


def shard_complete(index_root: Path, date_name: str) -> bool:
    """Check sidecars and the metadata marker written after index completion."""
    d = index_root / date_name
    return (d / "metadata.json").exists() and (d / "texts.parquet").exists()


def partials_dir(index_root: Path, date_name: str) -> Path:
    return index_root / date_name / "_partials"


def shard_partial_path(index_root: Path, date_name: str, shard_id: int) -> Path:
    return partials_dir(index_root, date_name) / f"shard_{shard_id:04d}.npz"


def progress_dir(index_root: Path) -> Path:
    return index_root / "_progress"


def progress_file(index_root: Path, shard_id: int) -> Path:
    return progress_dir(index_root) / f"shard_{shard_id:04d}.json"


def logs_dir(index_root: Path) -> Path:
    return index_root / "_logs"


def detect_num_gpus() -> int:
    try:
        import torch
        n = torch.cuda.device_count()
        if n > 0:
            return n
    except Exception:
        pass
    try:
        out = subprocess.check_output(["nvidia-smi", "-L"], text=True)
        return max(len([l for l in out.splitlines() if l.strip()]), 1)
    except Exception:
        return 1


def torchrun_env() -> dict[str, int] | None:
    """Read distributed ranks when RANK and WORLD_SIZE are set."""
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return None
    return {
        "rank": int(os.environ["RANK"]),
        "world_size": int(os.environ["WORLD_SIZE"]),
        "local_rank": int(os.environ.get("LOCAL_RANK", os.environ["RANK"])),
    }


def log(tag: str, msg: str) -> None:
    print(f"[{tag}] {msg}", flush=True)


def write_progress(
    index_root: Path,
    shard_id: int,
    *,
    phase: str,
    date: str = "",
    done: int = 0,
    total: int = 0,
) -> None:
    """Atomically publish this worker's current phase/progress to disk."""
    pdir = progress_dir(index_root)
    pdir.mkdir(parents=True, exist_ok=True)
    payload = {
        "shard": shard_id,
        "phase": phase,
        "date": date,
        "done": int(done),
        "total": int(total),
        "ts": time.time(),
    }
    target = progress_file(index_root, shard_id)
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload))
    tmp.replace(target)


def render_progress_loop(
    index_root: Path,
    num_shards: int,
    stop_event: threading.Event,
) -> None:
    """Launcher thread: one tqdm bar per shard, updated from progress files."""
    bars: list[tqdm] = []
    for i in range(num_shards):
        bars.append(tqdm(
            total=1,
            position=i,
            desc=f"shard {i} [waiting]",
            unit="doc",
            dynamic_ncols=True,
            leave=True,
            bar_format="{desc}: {percentage:5.1f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        ))

    try:
        while not stop_event.is_set():
            for i in range(num_shards):
                f = progress_file(index_root, i)
                if not f.exists():
                    continue
                try:
                    data = json.loads(f.read_text())
                except Exception:
                    continue

                phase = data.get("phase", "")
                date = data.get("date", "")
                done = int(data.get("done", 0))
                total = int(data.get("total", 0))

                bar = bars[i]
                desc = f"shard {i} [{phase}]"
                if date:
                    desc += f" {date}"
                bar.set_description(desc, refresh=False)
                if total > 0:
                    if bar.total != total:
                        bar.total = total
                    bar.n = done
                bar.refresh()
            time.sleep(RENDER_POLL_SEC)
    finally:
        for bar in bars:
            bar.close()


def normalize_column(df: pd.DataFrame, col: str) -> np.ndarray:
    if col not in df.columns:
        return np.array([""] * len(df), dtype=object)
    s = df[col].fillna("").astype(str).str.strip()
    s = s.where(s.str.lower() != "none", "")
    return s.values.astype(object)


def build_doc_records(
    feather_files: list[Path],
    shard_id: int = 0,
    num_shards: int = 1,
) -> dict[str, np.ndarray]:
    """Return aligned article arrays for this rank's round-robin row slice."""
    empty_keys = ("doc_texts", "titles", "summaries", "contents", "article_ids",
                  "urls", "source_files", "published_dates", "date_crawleds",
                  "date_publisheds", "date_modifieds", "date_sources")
    if not feather_files:
        return {k: np.array([], dtype=object) for k in empty_keys}

    dfs, src = [], []
    for f in feather_files:
        df = pd.read_feather(f)
        dfs.append(df)
        src.append(np.full(len(df), f.name, dtype=object))

    df = pd.concat(dfs, ignore_index=True)
    sources = np.concatenate(src) if src else np.array([], dtype=object)

    # A shared file order makes round-robin row sharding deterministic.
    if num_shards > 1:
        idx = np.arange(len(df))
        keep_rows = (idx % num_shards) == shard_id
        df = df.iloc[keep_rows].reset_index(drop=True)
        sources = sources[keep_rows]

    titles = normalize_column(df, "title")
    summaries_col = normalize_column(df, "summary")
    contents = normalize_column(df, "text")
    article_ids = normalize_column(df, "id")
    urls = normalize_column(df, "url")
    # The date column contains the resolved closest-to-crawl UTC day.

    published_dates = normalize_column(df, "date")
    date_crawleds = normalize_column(df, "date_crawled")
    date_publisheds = normalize_column(df, "date_published")
    date_modifieds = normalize_column(df, "date_modified")
    date_sources = normalize_column(df, "date_source")

    fallback = np.array([Path(s).stem for s in sources], dtype=object)
    final_titles = np.where(titles == "", fallback, titles)

    n = len(df)
    doc_texts = np.empty(n, dtype=object)
    for i in range(n):
        title = titles[i]
        summary = summaries_col[i]
        content = contents[i]

        sec: list[str] = []
        if summary:
            sec.extend(["- summary", summary, ""])
        if content:
            sec.extend(["- content", content])
        sections = "\n".join(sec).strip()

        parts: list[str] = []
        if title:
            parts.append(title)
        if sections:
            parts.append(sections)
        doc_texts[i] = "\n\n".join(parts).strip()

    keep = np.array([bool(t) for t in doc_texts])
    return {
        "doc_texts": doc_texts[keep],
        "titles": final_titles[keep],
        "summaries": summaries_col[keep],
        "contents": contents[keep],
        "article_ids": article_ids[keep],
        "urls": urls[keep],
        "source_files": sources[keep],
        "published_dates": published_dates[keep],
        "date_crawleds": date_crawleds[keep],
        "date_publisheds": date_publisheds[keep],
        "date_modifieds": date_modifieds[keep],
        "date_sources": date_sources[keep],
    }


def assigned_files(date_path: Path, shard_id: int, num_shards: int) -> list[Path]:
    """List all input feathers; build_doc_records partitions their rows by rank."""
    if date_path.is_file():
        return [date_path]
    return sorted(date_path.glob("*.feather"))


def run_worker(args: argparse.Namespace) -> None:
    from vllm import LLM

    data_root = Path(args.data_root)
    index_root = Path(args.index_root)
    date_paths = [resolve_date_path(data_root, d) or (data_root / d) for d in args.date]

    workload: list[tuple[Path, list[Path]]] = []
    for date_path in date_paths:
        files = assigned_files(date_path, args.shard_id, args.num_shards)
        if files:
            workload.append((date_path, files))

    tag = f"shard {args.shard_id}"
    if not workload:
        write_progress(index_root, args.shard_id, phase="idle", done=0, total=0)
        log(tag, "no input files for any date, exiting")
        return

    write_progress(index_root, args.shard_id, phase="loading model")
    log(tag, f"loading model on CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','?')}")
    # A pooling request must fit within one scheduler step.
    max_batched = max(MAX_NUM_BATCHED_TOKENS, args.max_tokens)
    llm = LLM(
        model=args.embedding_model,
        runner="pooling",
        trust_remote_code=True,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        max_model_len=args.max_tokens,
        max_num_seqs=MAX_NUM_SEQS,
        max_num_batched_tokens=max_batched,
    )
    log(tag, f"vLLM scheduler: max_num_seqs={MAX_NUM_SEQS} "
             f"max_num_batched_tokens={max_batched} max_model_len={args.max_tokens}")
    write_progress(index_root, args.shard_id, phase="ready")
    log(tag, "model loaded")

    for date_path, files in workload:
        embed_shard(
            llm=llm,
            date_dir=date_path,
            feather_files=files,
            index_root=index_root,
            shard_id=args.shard_id,
            num_shards=args.num_shards,
            batch_size=args.batch_size,
            max_tokens=args.max_tokens,
        )

    write_progress(index_root, args.shard_id, phase="done")


def embed_shard(
    llm,
    date_dir: Path,
    feather_files: list[Path],
    index_root: Path,
    shard_id: int,
    num_shards: int,
    batch_size: int,
    max_tokens: int,
) -> None:
    tag = f"shard {shard_id}"
    date_name = date_label(date_dir)

    write_progress(index_root, shard_id, phase="reading", date=date_name)
    log(tag, f"{date_name}: reading {len(feather_files)} feather file(s) "
             f"(row stride {shard_id}/{num_shards})")
    records = build_doc_records(feather_files, shard_id=shard_id, num_shards=num_shards)
    n = len(records["doc_texts"])
    if n == 0:
        log(tag, f"{date_name}: no valid docs in this shard's row slice, skipping")
        return

    texts = records["doc_texts"].tolist()
    # Leave one token of scheduler headroom and preserve special tokens.
    tokenizer = llm.get_tokenizer()
    truncate_to = max_tokens - 1

    log(tag, f"{date_name}: embedding {n:,} docs (batch={batch_size}, max_tokens={max_tokens})")
    os.environ["TOKENIZERS_PARALLELISM"] = "true"  # vLLM initialization may reset this.
    all_vecs: list[np.ndarray] = []
    token_lengths: list[int] = []
    use_local_bar = sys.stdout.isatty()
    pbar = tqdm(
        total=n,
        desc=f"{tag} {date_name}",
        unit="doc",
        dynamic_ncols=True,
        disable=not use_local_bar,
    )
    done = 0
    write_progress(index_root, shard_id, phase="embedding", date=date_name, done=0, total=n)
    try:
        for start in range(0, n, batch_size):
            batch_texts = texts[start:start + batch_size]
            batch_prompts = []
            for t in batch_texts:
                # Record full lengths; tokenizer truncation preserves the pooling token.
                full_ids = tokenizer.encode(t, add_special_tokens=True)
                token_lengths.append(len(full_ids))
                if len(full_ids) > truncate_to:
                    ids = tokenizer.encode(
                        t, truncation=True, max_length=truncate_to, add_special_tokens=True,
                    )
                else:
                    ids = full_ids
                batch_prompts.append({"prompt_token_ids": ids})
            outputs = llm.embed(batch_prompts)
            vecs = np.array([o.outputs.embedding for o in outputs], dtype=np.float32)
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            all_vecs.append(vecs / norms)
            done += len(batch_texts)
            pbar.update(len(batch_texts))
            write_progress(index_root, shard_id, phase="embedding", date=date_name, done=done, total=n)
    finally:
        pbar.close()

    vectors = np.concatenate(all_vecs, axis=0)

    write_progress(index_root, shard_id, phase="saving partial", date=date_name, done=n, total=n)
    pdir = partials_dir(index_root, date_name)
    pdir.mkdir(parents=True, exist_ok=True)
    np.savez(
        shard_partial_path(index_root, date_name, shard_id),
        vectors=vectors,
        titles=records["titles"],
        summaries=records["summaries"],
        contents=records["contents"],
        article_ids=records["article_ids"],
        urls=records["urls"],
        source_files=records["source_files"],
        published_dates=records["published_dates"],
        date_crawleds=records["date_crawleds"],
        date_publisheds=records["date_publisheds"],
        date_modifieds=records["date_modifieds"],
        date_sources=records["date_sources"],
        token_lengths=np.asarray(token_lengths, dtype=np.int64),
        truncate_to=np.int64(truncate_to),
    )
    n_trunc = int(sum(1 for L in token_lengths if L > truncate_to))
    log(tag, f"{date_name}: wrote partial ({n:,} docs, dim={vectors.shape[1]}, "
             f"truncated={n_trunc}/{n})")


def spawn_workers(args: argparse.Namespace, date_dirs: list[Path], num_gpus: int) -> None:
    index_root = Path(args.index_root)
    ldir = logs_dir(index_root)
    ldir.mkdir(parents=True, exist_ok=True)

    common = [
        sys.executable, os.path.abspath(__file__),
        "--role", "worker",
        "--num-shards", str(num_gpus),
        "--embedding-model", args.embedding_model,
        "--data-root", args.data_root,
        "--index-root", args.index_root,
        "--batch-size", str(args.batch_size),
        "--max-tokens", str(args.max_tokens),
        "--date", *[date_label(d) for d in date_dirs],
    ]

    # Separate node-local compile caches prevent worker rename races.
    cache_base = Path(os.environ.get("EMBED_CACHE_BASE", "/tmp/vllm_cache"))
    cache_base.mkdir(parents=True, exist_ok=True)

    procs: list[tuple[int, subprocess.Popen, Any]] = []
    for i in range(num_gpus):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(i)
        env["VLLM_CACHE_ROOT"] = str(cache_base / f"{os.uname().nodename}_gpu{i}")
        cmd = common + ["--shard-id", str(i)]
        log_file = open(ldir / f"shard_{i:04d}.log", "w")
        log("launcher", f"spawning worker {i} on GPU {i} (log: {ldir / f'shard_{i:04d}.log'})")
        proc = subprocess.Popen(cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT)
        procs.append((i, proc, log_file))

    stop_event = threading.Event()
    renderer = threading.Thread(
        target=render_progress_loop,
        args=(index_root, num_gpus, stop_event),
        daemon=True,
    )
    renderer.start()

    failed: list[tuple[int, int]] = []
    try:
        for i, p, lf in procs:
            rc = p.wait()
            lf.close()
            if rc != 0:
                failed.append((i, rc))
    except KeyboardInterrupt:
        log("launcher", "interrupted — terminating workers")
        for _, p, _ in procs:
            if p.poll() is None:
                p.terminate()
        raise
    finally:
        stop_event.set()
        renderer.join(timeout=2.0)
        print("\n" * (num_gpus + 1), flush=True)

    if failed:
        for i, rc in failed:
            print(
                f"[launcher] ERROR: worker {i} exit code {rc} — see {ldir / f'shard_{i:04d}.log'}",
                file=sys.stderr,
            )
        sys.exit(1)


def merge_date(
    index_root: Path,
    date_name: str,
    num_shards: int,
    embedding_model: str,
) -> dict[str, Any]:
    pdir = partials_dir(index_root, date_name)
    shard_files = sorted(pdir.glob("shard_*.npz")) if pdir.exists() else []
    if not shard_files:
        log("merge", f"{date_name}: no partials, skipping")
        return {"date": date_name, "status": "no_partials", "n_docs": 0, "n_files": 0}

    all_vecs: list[np.ndarray] = []
    titles, summaries, contents, ids, urls, sources, pub_dates = [], [], [], [], [], [], []
    crawled_dates, published_dates, modified_dates, date_sources = [], [], [], []
    token_lengths: list[int] = []
    truncate_to = 0
    for sf in shard_files:
        # Close each archive before removing partial files, including on NFS.
        with np.load(sf, allow_pickle=True) as data:
            all_vecs.append(data["vectors"])
            titles.extend(data["titles"].tolist())
            summaries.extend(data["summaries"].tolist())
            n_rows = len(data["titles"])
            if "contents" in data.files:
                contents.extend(data["contents"].tolist())
            else:
                contents.extend([""] * n_rows)
            ids.extend(data["article_ids"].tolist())
            urls.extend(data["urls"].tolist())
            sources.extend(data["source_files"].tolist())
            if "published_dates" in data.files:
                pub_dates.extend(data["published_dates"].tolist())
            else:
                pub_dates.extend([""] * n_rows)
            # Missing optional columns retain the same sidecar schema.
            for key, sink in (("date_crawleds", crawled_dates),
                              ("date_publisheds", published_dates),
                              ("date_modifieds", modified_dates),
                              ("date_sources", date_sources)):
                if key in data.files:
                    sink.extend(data[key].tolist())
                else:
                    sink.extend([""] * n_rows)
            if "token_lengths" in data.files:
                token_lengths.extend(data["token_lengths"].tolist())
            if "truncate_to" in data.files:
                truncate_to = int(data["truncate_to"])

    vectors = np.concatenate(all_vecs, axis=0)


    # Monthly search requires contiguous date ranges in day_offsets.
    day_offsets: dict[str, list[int]] = {}
    if is_month_label(date_name):
        days = np.array([(p or "")[:10] for p in pub_dates], dtype=object)
        # Exclude invalid or out-of-month dates before constructing row ranges.
        valid = np.array([bool(_DAY_RE.match(d)) and d[:7] == date_name for d in days])
        n_bad = int((~valid).sum())
        if n_bad:
            log("merge", f"{date_name}: dropping {n_bad} row(s) with a publish day "
                         f"outside {date_name} or unparseable")
        # Stable sorting preserves within-day order.
        order = np.argsort(days[valid], kind="stable")
        keep_idx = np.flatnonzero(valid)[order]
        vectors = vectors[keep_idx]
        _take = lambda seq: [seq[i] for i in keep_idx]
        titles, summaries, contents = _take(titles), _take(summaries), _take(contents)
        ids, urls, sources = _take(ids), _take(urls), _take(sources)
        pub_dates = _take(pub_dates)
        crawled_dates, published_dates = _take(crawled_dates), _take(published_dates)
        modified_dates, date_sources = _take(modified_dates), _take(date_sources)
        if token_lengths:
            token_lengths = _take(token_lengths)
        # Each day occupies the half-open row range [lo, hi).
        sorted_days = days[keep_idx]
        uniq, starts, counts = np.unique(sorted_days, return_index=True, return_counts=True)
        day_offsets = {str(d): [int(s), int(s + c)] for d, s, c in zip(uniq, starts, counts)}

    n_docs, dim = vectors.shape

    index = faiss.IndexFlatIP(dim)
    index.add(np.ascontiguousarray(vectors))

    out_dir = index_root / date_name
    out_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out_dir / "faiss.index"))

    text_truncated, _scrape_tok_lengths = _precompute_truncated(
        [t if t is not None else "" for t in contents], SCRAPE_TOKEN_BUDGET
    )

    # Keep article text and date metadata aligned with FAISS rows.
    pd.DataFrame(
        {
            "title": titles,
            "summary": summaries,
            "text": contents,
            "text_truncated": text_truncated,
            "article_id": ids,
            "url": urls,
            "source_file": sources,
            "published_date": pub_dates,
            "date_crawled": crawled_dates,
            "date_published": published_dates,
            "date_modified": modified_dates,
            "date_source": date_sources,
        }
    ).to_parquet(out_dir / "texts.parquet", compression="snappy", index=False)


    if token_lengths:
        lens = np.asarray(token_lengths, dtype=np.int64)
        cap = truncate_to or int(lens.max())
        n_trunc = int((lens > cap).sum())
        pct = lambda p: float(np.percentile(lens, p)) if lens.size else 0.0
        trunc_stats: dict[str, Any] = {
            "truncate_to_tokens": int(cap),
            "n_docs": int(lens.size),
            "n_truncated": n_trunc,
            "frac_truncated": round(n_trunc / lens.size, 6),
            "tokens_max": int(lens.max()),
            "tokens_mean": round(float(lens.mean()), 1),
            "tokens_p50": pct(50),
            "tokens_p90": pct(90),
            "tokens_p99": pct(99),
            "tokens_dropped_total": int(np.clip(lens - cap, 0, None).sum()),
        }
        log("merge",
            f"{date_name}: truncated {n_trunc}/{lens.size} docs "
            f"({100 * n_trunc / lens.size:.3f}%) over {cap} tok "
            f"[p50={trunc_stats['tokens_p50']:.0f} p99={trunc_stats['tokens_p99']:.0f} "
            f"max={trunc_stats['tokens_max']}]")
    else:
        trunc_stats = {"truncate_to_tokens": int(truncate_to), "n_docs": int(n_docs),
                       "n_truncated": 0, "frac_truncated": 0.0}

    metadata = {
        "date": date_name,
        "n_docs": int(n_docs),
        "n_files": len(set(sources)),
        "dim": int(dim),
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "embedding_model": embedding_model,
        "num_shards": num_shards,
        "truncation": trunc_stats,
        # Serving validates this budget and encoding before using cached text.
        "scrape_truncation": {
            "column": "text_truncated",
            # Store the encoding name rather than its model alias.
            "tokenizer": _scrape_encoding_name(),
            "token_budget": SCRAPE_TOKEN_BUDGET,
        },
    }
    if day_offsets:
        # Both fields are required for date-restricted monthly retrieval.
        metadata["row_order"] = "published_date"
        metadata["day_offsets"] = day_offsets
        metadata["n_days"] = len(day_offsets)
    else:

        # Daily shards also retain inline metadata for compatibility.
        metadata.update({
            "titles": titles,
            "summaries": summaries,
            "article_ids": ids,
            "urls": urls,
            "source_files": sources,
        })
    # Write the completion marker last for --skip-existing.
    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    shutil.rmtree(pdir)
    log("merge", f"{date_name}: {n_docs:,} docs, dim={dim} -> {out_dir}")
    return {
        "date": date_name,
        "status": "ok",
        "n_docs": int(n_docs),
        "n_files": len(set(sources)),
        "n_truncated": int(trunc_stats.get("n_truncated", 0)),
        "frac_truncated": float(trunc_stats.get("frac_truncated", 0.0)),
    }


def run_torchrun(args: argparse.Namespace, env: dict[str, int]) -> None:
    """Embed one rank's rows, then synchronize around the rank-0 merge."""
    import torch.distributed as dist

    rank = env["rank"]
    world_size = env["world_size"]
    local_rank = env["local_rank"]
    tag = f"rank {rank}/{world_size}"

    os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)

    data_root = Path(args.data_root)
    index_root = Path(args.index_root)

    if not data_root.exists():
        print(f"[{tag}] ERROR: data root not found: {data_root}", file=sys.stderr)
        sys.exit(1)

    if args.date:
        date_dirs = []
        missing = []
        for d in args.date:
            p = resolve_date_path(data_root, d)
            if p is None:
                missing.append(d)
            else:
                date_dirs.append(p)
        if missing:
            for m in missing:
                print(f"[{tag}] ERROR: no <{m}>.feather or <{m}>/ under {data_root}", file=sys.stderr)
            sys.exit(1)
    else:
        date_dirs = list_date_dirs(data_root)

    if args.skip_existing:
        before = len(date_dirs)
        date_dirs = [d for d in date_dirs if not shard_complete(index_root, date_label(d))]
        if rank == 0:
            log("launcher", f"skip-existing: {before - len(date_dirs)} skipped, {len(date_dirs)} to build")

    # Rank 0 clears shared state before all ranks cross the barrier.
    if rank == 0:
        log("launcher", f"data root      : {data_root}")
        log("launcher", f"index root     : {index_root}")
        log("launcher", f"embedding model: {args.embedding_model}")
        log("launcher", f"date dirs      : {[date_label(d) for d in date_dirs]}")
        log("launcher", f"world size     : {world_size}")
        log("launcher", f"batch size     : {args.batch_size}")
        for d in date_dirs:
            pd_ = partials_dir(index_root, date_label(d))
            if pd_.exists():
                shutil.rmtree(pd_)
        pdir = progress_dir(index_root)
        if pdir.exists():
            shutil.rmtree(pdir)

    # Synchronization needs CPU barriers only.
    dist.init_process_group(backend="gloo")
    dist.barrier()

    # Prevent vLLM subprocesses from joining the existing torchrun group.
    for var in (
        "RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE",
        "MASTER_ADDR", "MASTER_PORT", "GROUP_RANK", "ROLE_RANK",
        "ROLE_NAME", "ROLE_WORLD_SIZE",
        "TORCHELASTIC_RESTART_COUNT", "TORCHELASTIC_MAX_RESTARTS",
        "TORCHELASTIC_RUN_ID", "TORCHELASTIC_USE_AGENT_STORE",
        "TORCHELASTIC_ERROR_FILE",
    ):
        os.environ.pop(var, None)

    if not date_dirs:
        if rank == 0:
            log("launcher", "nothing to do")
        dist.barrier()
        dist.destroy_process_group()
        return

    worker_args = argparse.Namespace(
        role="worker",
        shard_id=rank,
        num_shards=world_size,
        embedding_model=args.embedding_model,
        data_root=args.data_root,
        index_root=args.index_root,
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
        date=[date_label(d) for d in date_dirs],
        num_gpus=1,
        skip_existing=False,
    )
    run_worker(worker_args)

    # All partials must be written before rank 0 merges.
    dist.barrier()

    if rank == 0:
        log("launcher", "merging shard partials")
        results = [
            merge_date(index_root, date_label(d), world_size, args.embedding_model)
            for d in date_dirs
        ]
        summary = {
            "built_at": datetime.now().isoformat(timespec="seconds"),
            "embedding_model": args.embedding_model,
            "world_size": world_size,
            "batch_size": args.batch_size,
            "total_date_dirs": len(date_dirs),
            "ok_date_dirs": sum(1 for r in results if r["status"] == "ok"),
            "total_docs": sum(int(r["n_docs"]) for r in results),
            "results": results,
        }
        index_root.mkdir(parents=True, exist_ok=True)
        (index_root / "build_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        pdir = progress_dir(index_root)
        if pdir.exists():
            shutil.rmtree(pdir)
        log("launcher", f"done. summary: {index_root / 'build_summary.json'}")

    # Keep other ranks alive until merge completes.
    dist.barrier()
    dist.destroy_process_group()


def run_launcher(args: argparse.Namespace) -> None:
    data_root = Path(args.data_root)
    index_root = Path(args.index_root)

    if not data_root.exists():
        print(f"ERROR: data root not found: {data_root}", file=sys.stderr)
        sys.exit(1)

    if args.date:
        date_dirs = []
        missing = []
        for d in args.date:
            p = resolve_date_path(data_root, d)
            if p is None:
                missing.append(d)
            else:
                date_dirs.append(p)
        if missing:
            for m in missing:
                print(f"ERROR: no <{m}>.feather or <{m}>/ under {data_root}", file=sys.stderr)
            sys.exit(1)
    else:
        date_dirs = list_date_dirs(data_root)

    if args.skip_existing:
        before = len(date_dirs)
        date_dirs = [d for d in date_dirs if not shard_complete(index_root, date_label(d))]
        log("launcher", f"skip-existing: {before - len(date_dirs)} skipped, {len(date_dirs)} to build")

    if not date_dirs:
        log("launcher", "nothing to do")
        return

    num_gpus = args.num_gpus if args.num_gpus > 0 else detect_num_gpus()

    log("launcher", f"data root      : {data_root}")
    log("launcher", f"index root     : {index_root}")
    log("launcher", f"embedding model: {args.embedding_model}")
    log("launcher", f"date dirs      : {[date_label(d) for d in date_dirs]}")
    log("launcher", f"num GPUs       : {num_gpus}")
    log("launcher", f"batch size     : {args.batch_size}")

    for d in date_dirs:
        pd_ = partials_dir(index_root, date_label(d))
        if pd_.exists():
            shutil.rmtree(pd_)
    pdir = progress_dir(index_root)
    if pdir.exists():
        shutil.rmtree(pdir)

    if num_gpus == 1:
        worker_args = argparse.Namespace(
            role="worker",
            shard_id=0,
            num_shards=1,
            embedding_model=args.embedding_model,
            data_root=args.data_root,
            index_root=args.index_root,
            batch_size=args.batch_size,
            max_tokens=args.max_tokens,
            date=[date_label(d) for d in date_dirs],
            num_gpus=1,
            skip_existing=False,
        )
        run_worker(worker_args)
    else:
        log("launcher", f"worker logs: {logs_dir(index_root)}/shard_NNNN.log (tail -F to watch raw output)")
        spawn_workers(args, date_dirs, num_gpus)

    log("launcher", "merging shard partials")
    results = [
        merge_date(index_root, date_label(d), num_gpus, args.embedding_model)
        for d in date_dirs
    ]

    summary = {
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "embedding_model": args.embedding_model,
        "num_gpus": num_gpus,
        "batch_size": args.batch_size,
        "total_date_dirs": len(date_dirs),
        "ok_date_dirs": sum(1 for r in results if r["status"] == "ok"),
        "total_docs": sum(int(r["n_docs"]) for r in results),
        "results": results,
    }
    index_root.mkdir(parents=True, exist_ok=True)
    (index_root / "build_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if pdir.exists():
        shutil.rmtree(pdir)
    log("launcher", f"done. summary: {index_root / 'build_summary.json'}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Data-parallel Qwen3-Embedding + FAISS index builder for CC News feathers."
    )
    p.add_argument("--role", choices=["launcher", "worker"], default="launcher",
                   help="Internal: launcher spawns workers; workers do the embedding.")
    p.add_argument("--embedding-model", type=str,
                   default=os.environ.get("EMBEDDING_MODEL_NAME", DEFAULT_EMBEDDING_MODEL))
    p.add_argument("--data-root", type=str, required=True)
    p.add_argument("--index-root", type=str, required=True)
    p.add_argument("--date", nargs="+", default=None,
                   help="Restrict to these date subdir names. Default: all subdirs of data-root.")
    p.add_argument("--num-gpus", type=int, default=0,
                   help="Number of GPUs / data-parallel workers. 0 = auto-detect.")
    p.add_argument("--batch-size", type=int, default=EMBED_BATCH_SIZE,
                   help=f"Docs per embed call (default {EMBED_BATCH_SIZE}).")
    p.add_argument("--max-tokens", type=int, default=MAX_TOKENS,
                   help=f"Truncate each doc to this many tokens (default {MAX_TOKENS}). "
                        "Qwen3-Embedding-8B supports up to 32000, but shorter is far faster "
                        "and usually fine for news.")
    p.add_argument("--shard-id", type=int, default=0, help="(worker only)")
    p.add_argument("--num-shards", type=int, default=1, help="(worker only)")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip date shards with completed metadata and text sidecars.")
    return p


def main() -> None:
    args = build_parser().parse_args()

    if args.role == "worker":
        run_worker(args)
        return

    env = torchrun_env()
    if env is not None:
        run_torchrun(args, env)
        return

    run_launcher(args)


if __name__ == "__main__":
    main()
