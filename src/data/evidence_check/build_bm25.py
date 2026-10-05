"""Build dated lexical indexes for evidence screening."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import bm25s
import pyarrow.parquet as pq
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

logger = logging.getLogger(__name__)


def _ts_now() -> str:
    """Return the UTC build timestamp."""
    import datetime as _dt
    return _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def _build_one_shard(args: tuple[str, Path, Path, bool]) -> dict[str, Any]:
    """Build one BM25 shard in a worker process and return its status."""
    shard_name, index_shard_dir, bm25_shard_dir, force = args
    parquet_path = index_shard_dir / "texts.parquet"
    meta_path = bm25_shard_dir / "meta.json"
    bm25_dir = bm25_shard_dir / "bm25s"

    if not parquet_path.exists():
        return {"shard": shard_name, "status": "skip_no_parquet"}
    if meta_path.exists() and bm25_dir.exists() and not force:
        return {"shard": shard_name, "status": "skip_exists"}

    table = pq.read_table(parquet_path, columns=["title", "text"])
    titles = table.column("title").to_pylist()
    texts = table.column("text").to_pylist()
    if len(titles) != len(texts):
        return {
            "shard": shard_name,
            "status": "error_misaligned",
            "n_titles": len(titles),
            "n_texts": len(texts),
        }
    n = len(texts)

    docs = [
        f"{t or ''} {x or ''}".strip()
        for t, x in zip(titles, texts)
    ]
    # Release shard-sized buffers before tokenization.
    del titles, texts, table

    tokens = bm25s.tokenize(
        docs, stopwords="en", lower=True, show_progress=False,
    )
    del docs

    retriever = bm25s.BM25(method="lucene")
    retriever.index(tokens, show_progress=False)

    bm25_shard_dir.mkdir(parents=True, exist_ok=True)
    retriever.save(str(bm25_dir))

    meta = {
        "shard": shard_name,
        "n_docs": n,
        "parquet_path": str(parquet_path),
        "method": "lucene",
        "stopwords": "en",
        "stemmer": None,
        "lower": True,
        "built_at": _ts_now(),
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    return {"shard": shard_name, "status": "ok", "n_docs": n}


def _list_shards(index_root: Path, names: list[str] | None = None) -> list[str]:
    """List corpus shards, or use the supplied names."""
    if names is not None:
        return list(names)
    return sorted(
        d.name for d in index_root.iterdir()
        if d.is_dir() and not d.name.startswith("_")
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--index-root", type=Path, required=True,
        help="Corpus root containing <shard>/texts.parquet.",
    )
    ap.add_argument(
        "--bm25-root", type=Path, required=True,
        help="Output root for the BM25 index. Created if missing.",
    )
    ap.add_argument(
        "--shards", nargs="*", default=None,
        help="Restrict to these shard names (default: all subdirs of "
             "--index-root that don't start with '_').",
    )
    ap.add_argument(
        "--workers", type=int, default=4,
        help="Concurrent shard builds; allow roughly 6-8 GB per 1.3M-document shard.",
    )
    ap.add_argument(
        "--force", action="store_true",
        help="Rebuild shards that already have a meta.json + bm25s dir.",
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

    if not args.index_root.is_dir():
        ap.error(f"--index-root does not exist: {args.index_root}")
    args.bm25_root.mkdir(parents=True, exist_ok=True)

    shards = _list_shards(args.index_root, args.shards)
    if not shards:
        logger.info("no shards to build under %s", args.index_root)
        return

    work = [
        (
            s,
            args.index_root / s,
            args.bm25_root / s,
            args.force,
        )
        for s in shards
    ]
    logger.info("building BM25 across %d shards (workers=%d, force=%s)",
                len(work), args.workers, args.force)

    n_ok = 0
    n_skip = 0
    n_err = 0

    # Processes parallelize CPU-bound tokenization.
    with ProcessPoolExecutor(max_workers=args.workers) as pool, \
         logging_redirect_tqdm(), \
         tqdm(
             total=len(work), unit="shard",
             desc=f"build BM25 (workers={args.workers})",
             dynamic_ncols=True,
         ) as bar:
        futs = [pool.submit(_build_one_shard, w) for w in work]
        for fut in as_completed(futs):
            try:
                res = fut.result()
            except Exception as e:
                logger.exception("worker raised: %s", e)
                n_err += 1
                bar.update(1)
                continue
            status = res.get("status", "")
            if status == "ok":
                n_ok += 1
                logger.info("[%s] built %d docs", res["shard"], res["n_docs"])
            elif status.startswith("skip"):
                n_skip += 1
            else:
                n_err += 1
                logger.warning("[%s] %s (details=%s)",
                               res.get("shard"), status, res)
            bar.set_postfix(ok=n_ok, skip=n_skip, err=n_err, refresh=False)
            bar.update(1)

    logger.info("BM25 build complete: ok=%d  skip=%d  err=%d  out=%s",
                n_ok, n_skip, n_err, args.bm25_root)


if __name__ == "__main__":
    sys.exit(main())
