"""Collect daily Polymarket price histories."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm


CLOB_PRICES_URL = "https://clob.polymarket.com/prices-history"
DEFAULT_TIMEOUT = 30
DEFAULT_FIDELITY_MIN = 60      # Minutes per history sample.
CHUNK_DAYS = 15                # Maximum days per history request.
DEFAULT_RATE_LIMIT = 12.0
DEFAULT_WORKERS = 6


logger = logging.getLogger("pm_crowd_belief")
logger.setLevel(logging.INFO)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s"))
    logger.addHandler(h)


class TokenBucket:
    def __init__(self, rps: float, burst: float):
        self.rate = float(rps)
        self.capacity = float(burst)
        self.tokens = float(burst)
        self.last = time.monotonic()
        self.lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            with self.lock:
                now = time.monotonic()
                self.tokens = min(
                    self.capacity, self.tokens + (now - self.last) * self.rate
                )
                self.last = now
                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    return
                wait = (1.0 - self.tokens) / self.rate
            time.sleep(wait)


_LIMITER: Optional[TokenBucket] = None


def _throttle() -> None:
    if _LIMITER is not None:
        _LIMITER.acquire()


def make_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=10,
        backoff_factor=1.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
        respect_retry_after_header=True,
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=32, pool_maxsize=32)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    s.headers.update({"User-Agent": "PMCrowdBelief/1.0"})
    return s


def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def cache_get(cache_dir: str, key: str) -> Any:
    if not cache_dir:
        return None
    path = os.path.join(cache_dir, f"{key}.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def cache_set(cache_dir: str, key: str, obj: Any) -> None:
    if not cache_dir:
        return
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{key}.json")
    tmp = f"{path}.tmp.{os.getpid()}.{threading.get_ident()}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
    os.replace(tmp, path)


def _parse_iso(x: Any) -> Optional[pd.Timestamp]:
    if x is None or x == "":
        return None
    try:
        return pd.to_datetime(x, utc=True)
    except Exception:
        return None


def _normalize_price(p: float) -> float:
    if p > 1.0 and p <= 100.0:
        p /= 100.0
    return float(np.clip(p, 0.0, 1.0))


def fetch_price_history(
    session: requests.Session,
    token_id: str,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    cache_dir: str,
) -> List[Tuple[pd.Timestamp, float]]:
    """Walk /prices-history in CHUNK_DAYS-day windows. Returns (ts, price) pairs."""
    out: List[Tuple[pd.Timestamp, float]] = []
    cur = int(start_ts.timestamp())
    final = int(end_ts.timestamp())
    step = CHUNK_DAYS * 86400

    while cur < final:
        nxt = min(cur + step, final)
        params = {
            "market": token_id,
            "startTs": cur,
            "endTs": nxt,
            "fidelity": DEFAULT_FIDELITY_MIN,
        }
        key = "prices_" + _sha1(json.dumps(params, sort_keys=True))
        cached = cache_get(cache_dir, key)
        if cached is None:
            _throttle()
            r = session.get(CLOB_PRICES_URL, params=params, timeout=DEFAULT_TIMEOUT)
            if r.status_code in (400, 404, 422):
                # Unavailable token windows contribute no history.
                cached = {"history": []}
            else:
                r.raise_for_status()
                cached = r.json()
            cache_set(cache_dir, key, cached)

        if isinstance(cached, dict):
            for pt in cached.get("history", []):
                try:
                    t = pd.Timestamp(int(pt["t"]), unit="s", tz="UTC")
                    p = _normalize_price(float(pt["p"]))
                    out.append((t, p))
                except Exception:
                    continue

        cur = nxt

    return out


def _to_daily(pairs: List[Tuple[pd.Timestamp, float]]) -> pd.Series:
    """Average within UTC days. Returns a Series indexed by midnight-UTC days."""
    if not pairs:
        return pd.Series(dtype=float)
    df = pd.DataFrame(pairs, columns=["ts", "p"]).dropna()
    if df.empty:
        return pd.Series(dtype=float)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.sort_values("ts").set_index("ts")
    df["p"] = df["p"].astype(float).clip(0.0, 1.0)
    s = df["p"].resample("1D").mean().dropna()
    s.index = pd.to_datetime(s.index, utc=True)
    return s


def _option_label(market: Dict[str, Any], fallback_idx: int) -> str:
    return (
        market.get("label")
        or market.get("question")
        or f"option_{fallback_idx}"
    )


def _option_window(
    market: Dict[str, Any],
    event_record: Dict[str, Any],
) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    """Return the trading window, preferring market open and recorded close times."""
    start = (
        _parse_iso(market.get("start_date"))
        or _parse_iso(market.get("created_at"))
        or _parse_iso(event_record.get("start_date"))
        or _parse_iso(event_record.get("creation_date"))
    )
    end = (
        _parse_iso(market.get("closed_time"))
        or _parse_iso(event_record.get("close_date"))
        or _parse_iso(market.get("end_date"))
        or _parse_iso(event_record.get("resolution_date"))
        or _parse_iso(event_record.get("end_date"))
    )
    return start, end


def _ensure_unique_label(label: str, existing: Dict[str, Any]) -> str:
    if label not in existing:
        return label
    i = 2
    while f"{label}__{i}" in existing:
        i += 1
    return f"{label}__{i}"


def compute_belief(
    event: Dict[str, Any],
    *,
    session: requests.Session,
    cache_dir: str,
    clip_start: Optional[pd.Timestamp],
    clip_end: Optional[pd.Timestamp],
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Return daily belief fields and a skip reason, if the event is unsupported."""
    neg_risk = bool(event.get("neg_risk", False))
    markets = event.get("markets") or []
    n = len(markets)

    if n == 0:
        return None, "no_markets"

    is_binary = (n == 1) and not neg_risk
    is_multi_neg = (n >= 2) and neg_risk
    if not (is_binary or is_multi_neg):
        return None, "unsupported_layout"

    options: List[Tuple[str, str, pd.Timestamp, pd.Timestamp]] = []
    seen_labels: Dict[str, int] = {}
    for i, m in enumerate(markets):
        token = m.get("yes_token_id")
        if not token:
            continue
        start, end = _option_window(m, event)
        if start is None or end is None or end <= start:
            continue
        if clip_start is not None and end < clip_start:
            continue
        if clip_end is not None and start > clip_end:
            continue
        eff_start = start if clip_start is None else max(start, clip_start)
        eff_end = end if clip_end is None else min(end, clip_end)
        if eff_end <= eff_start:
            continue

        label = _option_label(m, i)
        label = _ensure_unique_label(label, seen_labels)
        seen_labels[label] = i

        options.append((label, str(token), eff_start, eff_end))

    if not options:
        return None, "no_yes_tokens_or_window"

    yes_daily: Dict[str, pd.Series] = {}
    option_windows: Dict[str, Tuple[pd.Timestamp, pd.Timestamp]] = {}
    for label, token, s, e in options:
        pairs = fetch_price_history(session, token, s, e, cache_dir)
        yes_daily[label] = _to_daily(pairs)
        option_windows[label] = (s.normalize(), e.normalize())

    # Include price-gap days across the complete trading window.
    grid_start = min(s for s, _ in option_windows.values())
    grid_end = max(e for _, e in option_windows.values())
    if grid_end <= grid_start:
        return None, "empty_window"
    daily_index = pd.date_range(grid_start, grid_end, freq="1D", tz="UTC")

    # Missing values stay null, including dates outside an option's lifetime.
    yes_aligned = pd.DataFrame(
        {label: s.reindex(daily_index) for label, s in yes_daily.items()},
        index=daily_index,
    )

    total_days = len(daily_index)
    missingness = {
        label: {
            "days_total": total_days,
            "days_with_value": int(yes_aligned[label].notna().sum()),
            "days_missing": int(yes_aligned[label].isna().sum()),
        }
        for label in yes_aligned.columns
    }

    iso_index = [t.isoformat() for t in daily_index]

    def _ser(col: pd.Series) -> List[Optional[float]]:
        return [None if pd.isna(v) else float(v) for v in col.values]

    raw_yes_history = {label: _ser(yes_aligned[label]) for label in yes_aligned.columns}

    out: Dict[str, Any] = {
        "probability_start_date": iso_index[0],
        "probability_end_date": iso_index[-1],
        "daily_index": iso_index,
        "raw_yes_history": raw_yes_history,
        "missingness": missingness,
    }

    if is_binary:
        label = next(iter(yes_aligned.columns))
        no_series = 1.0 - yes_aligned[label]
        out["raw_no_history"] = {label: _ser(no_series)}
        out["belief_kind"] = "binary"
        out["normalized_history"] = None
        out["daily_probability_sum"] = None
        return out, None

    daily_sum = yes_aligned.sum(axis=1, skipna=True)
    # Zero-sum days remain undefined.
    zero_or_nan = (daily_sum.isna()) | (daily_sum <= 0)
    safe_sum = daily_sum.where(~zero_or_nan, np.nan)

    normalized = yes_aligned.divide(safe_sum, axis=0)
    normalized = normalized.mask(zero_or_nan, np.nan)

    out["belief_kind"] = "multi_neg_risk"
    out["daily_probability_sum"] = _ser(daily_sum)
    out["normalized_history"] = {
        label: _ser(normalized[label]) for label in normalized.columns
    }
    return out, None


def _read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            yield json.loads(ln)


def _load_done(path: str) -> set:
    if not os.path.exists(path):
        return set()
    with open(path, "r", encoding="utf-8") as f:
        return {ln.strip() for ln in f if ln.strip()}


def _append_done(path: str, eid: str) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(eid + "\n")


def _append_skip(path: str, eid: str, reason: str) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"event_id": eid, "reason": reason}) + "\n")


def run(
    *,
    input_path: str,
    output_path: str,
    cache_dir: str,
    rate_limit: float,
    workers: int,
    start_date: Optional[str],
    end_date: Optional[str],
    force: bool,
) -> int:
    global _LIMITER
    _LIMITER = TokenBucket(rps=rate_limit, burst=max(rate_limit * 1.5, 1.0))

    clip_start = _parse_iso(f"{start_date}T00:00:00Z") if start_date else None
    clip_end = _parse_iso(f"{end_date}T23:59:59Z") if end_date else None

    out_dir = os.path.dirname(os.path.abspath(output_path)) or "."
    os.makedirs(out_dir, exist_ok=True)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    done_path = output_path + ".done"
    skip_path = output_path + ".skipped.jsonl"

    done: set = set() if force else _load_done(done_path)
    if force and os.path.exists(output_path):
        os.remove(output_path)
    if force and os.path.exists(done_path):
        os.remove(done_path)
    if force and os.path.exists(skip_path):
        os.remove(skip_path)

    total = sum(1 for _ in _read_jsonl(input_path))

    session = make_session()
    n_written = 0
    n_skipped = 0
    write_lock = threading.Lock()

    out_mode = "a" if (not force and os.path.exists(output_path)) else "w"
    out_f = open(output_path, out_mode, encoding="utf-8")
    bar = tqdm(total=total, desc="Events", unit="ev")

    def _process(event: Dict[str, Any]) -> Tuple[str, Optional[Dict[str, Any]], str]:
        eid = str(event.get("event_id"))
        try:
            belief, reason = compute_belief(
                event,
                session=session,
                cache_dir=cache_dir,
                clip_start=clip_start,
                clip_end=clip_end,
            )
        except Exception as e:
            return eid, None, f"exception:{type(e).__name__}:{e}"
        if belief is None:
            return eid, None, reason or "unknown"
        merged = dict(event)
        merged.update(belief)
        return eid, merged, ""

    try:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=workers) as ex:
            pending = []
            for event in _read_jsonl(input_path):
                eid = str(event.get("event_id"))
                if eid in done:
                    bar.update(1)
                    continue
                pending.append(ex.submit(_process, event))

            for fut in as_completed(pending):
                bar.update(1)
                try:
                    eid, merged, reason = fut.result()
                except Exception as e:
                    logger.warning("worker crashed: %s", e)
                    n_skipped += 1
                    continue
                if merged is None:
                    with write_lock:
                        _append_skip(skip_path, eid, reason)
                    logger.warning("event %s skipped: %s", eid, reason)
                    n_skipped += 1
                    continue
                with write_lock:
                    out_f.write(json.dumps(merged, ensure_ascii=False) + "\n")
                    out_f.flush()
                    _append_done(done_path, eid)
                n_written += 1
                bar.set_postfix(written=n_written, skipped=n_skipped)
    finally:
        out_f.close()
        bar.close()

    logger.info("Wrote %d records, skipped %d -> %s", n_written, n_skipped, output_path)
    return n_written


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Construct daily crowd-belief trajectories for Polymarket events."
    )
    p.add_argument("--input", required=True, help="Event metadata JSONL from data.collect_events.")
    p.add_argument("--output", required=True, help="Output JSONL")
    p.add_argument(
        "--start-date",
        default=None,
        help=(
            "Inclusive YYYY-MM-DD lower bound on the market trading window."
        ),
    )
    p.add_argument(
        "--end-date",
        default=None,
        help=(
            "Inclusive YYYY-MM-DD upper bound on the market trading window."
        ),
    )
    p.add_argument(
        "--cache-dir",
        default=os.getenv("PM_BELIEF_CACHE_DIR", "pm_belief_cache"),
        help="Directory for chunked price-history cache. Empty string disables caching.",
    )
    p.add_argument(
        "--rate-limit",
        type=float,
        default=DEFAULT_RATE_LIMIT,
        help=f"Outbound HTTP rps cap (default {DEFAULT_RATE_LIMIT})",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=(
            "Concurrent event workers sharing the global rate limit (default: 6)."
        ),
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Recompute everything, ignoring .done sidecar.",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    return 0 if run(
        input_path=args.input,
        output_path=args.output,
        cache_dir=args.cache_dir,
        rate_limit=args.rate_limit,
        workers=args.workers,
        start_date=args.start_date,
        end_date=args.end_date,
        force=args.force,
    ) >= 0 else 1


if __name__ == "__main__":
    sys.exit(main())
