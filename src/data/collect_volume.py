"""Collect daily Polymarket trading activity."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import threading
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm


TRADES_URL = "https://data-api.polymarket.com/trades"
DEFAULT_TIMEOUT = 30
DEFAULT_RATE_LIMIT = 8.0
DEFAULT_WORKERS = 4
PAGE_LIMIT = 1000          # REST page limit.
HARD_OFFSET_CAP = 3000     # Larger REST offsets return HTTP 400.

# Goldsky indexes CTF exchange fills with timestamp and cursor filters.
SUBGRAPH_URL = (
    "https://api.goldsky.com/api/public/project_cl6mb8i9h0003e201j6li0diw"
    "/subgraphs/orderbook-subgraph/0.0.1/gn"
)
# Count disjoint maker-USDC and taker-USDC fills.
SUBGRAPH_PAGE_SIZE = 1000
USDC_ASSET_ID = "0"
TOKEN_DECIMALS = 1_000_000  # Both asset types use six decimal places.


logger = logging.getLogger("pm_volume")
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
        allowed_methods=frozenset(["GET", "POST"]),
        respect_retry_after_header=True,
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=32, pool_maxsize=32)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    s.headers.update({"User-Agent": "PMVolume/1.0"})
    return s


def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def cache_get(cache_dir: str, key: str) -> Any:
    if not cache_dir:
        return None
    path = os.path.join(cache_dir, key[:2], f"{key}.json")
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
    sub = os.path.join(cache_dir, key[:2])
    os.makedirs(sub, exist_ok=True)
    path = os.path.join(sub, f"{key}.json")
    tmp = f"{path}.tmp.{os.getpid()}.{threading.get_ident()}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
    os.replace(tmp, path)


def _parse_iso_dt(s: Optional[str]) -> Optional[datetime]:
    if not s or not isinstance(s, str):
        return None
    try:
        s2 = s.replace("Z", "+00:00")
        dt = datetime.fromisoformat(s2)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _day_str(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d")


def _day_str_from_ts(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")


def _fetch_trades_page(
    session: requests.Session,
    *,
    condition_id: str,
    offset: int,
    limit: int,
    cache_dir: str,
) -> Optional[List[Dict[str, Any]]]:
    """Fetch one /trades page newest-first. Returns list, or None on fatal error."""
    params = {"market": condition_id, "limit": limit, "offset": offset}
    key = "trades_" + _sha1(json.dumps(params, sort_keys=True))
    cached = cache_get(cache_dir, key)
    if cached is not None:
        return cached if isinstance(cached, list) else None

    _throttle()
    r = session.get(TRADES_URL, params=params, timeout=DEFAULT_TIMEOUT)
    if r.status_code == 400:
        # Offset-limit failures are not cached.
        return None
    if r.status_code != 200:
        r.raise_for_status()
    data = r.json()
    if not isinstance(data, list):
        data = []
    cache_set(cache_dir, key, data)
    return data


def crawl_market_trades(
    session: requests.Session,
    *,
    condition_id: str,
    window_start: datetime,
    window_end: datetime,
    cache_dir: str,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Any]]:
    """Aggregate REST trades in [window_start, window_end); report truncation."""
    daily: Dict[str, Dict[str, float]] = {}
    n_trades = 0
    n_pages = 0
    offset = 0
    truncated = False
    oldest_ts: Optional[int] = None
    stop_after_window_seen = False

    win_start_ts = int(window_start.timestamp())
    win_end_ts = int(window_end.timestamp())

    while True:
        if offset > HARD_OFFSET_CAP:
            if not stop_after_window_seen:
                truncated = True
            break

        page_lim = min(PAGE_LIMIT, max(1, HARD_OFFSET_CAP - offset + PAGE_LIMIT))
        page_lim = min(page_lim, PAGE_LIMIT)

        page = _fetch_trades_page(
            session,
            condition_id=condition_id,
            offset=offset,
            limit=page_lim,
            cache_dir=cache_dir,
        )
        if page is None:
            if not stop_after_window_seen:
                truncated = True
            break

        n_pages += 1
        if not page:
            break

        for t in page:
            try:
                ts = int(t.get("timestamp", 0))
                size = float(t.get("size", 0))
                price = float(t.get("price", 0))
            except (TypeError, ValueError):
                continue
            if ts <= 0:
                continue
            if oldest_ts is None or ts < oldest_ts:
                oldest_ts = ts
            if win_start_ts <= ts < win_end_ts:
                d = _day_str_from_ts(ts)
                slot = daily.setdefault(
                    d, {"trades": 0, "share_volume": 0.0, "notional": 0.0}
                )
                slot["trades"] += 1
                slot["share_volume"] += size
                slot["notional"] += size * price
                n_trades += 1

        try:
            last_ts = int(page[-1].get("timestamp", 0))
        except (TypeError, ValueError):
            last_ts = 0

        if last_ts and last_ts < win_start_ts:
            stop_after_window_seen = True
            break
        if len(page) < page_lim:
            break

        offset += page_lim

    stats = {
        "condition_id": condition_id,
        "n_trades": n_trades,
        "n_pages": n_pages,
        "truncated": truncated,
        "oldest_trade": (
            datetime.fromtimestamp(oldest_ts, tz=timezone.utc).isoformat()
            if oldest_ts is not None
            else None
        ),
    }
    return daily, stats


# Maker/taker queries partition fills by the side holding USDC.
_SUBGRAPH_QUERY_MAKER = """
query FillsMaker($tokens: [String!]!, $startTs: BigInt!, $endTs: BigInt!,
                 $cursorId: ID!, $size: Int!) {
  rows: orderFilledEvents(
    first: $size,
    orderBy: id, orderDirection: asc,
    where: {
      makerAssetId: "0",
      takerAssetId_in: $tokens,
      timestamp_gte: $startTs, timestamp_lt: $endTs,
      id_gt: $cursorId
    }
  ) {
    id timestamp makerAmountFilled takerAmountFilled
  }
}
"""

_SUBGRAPH_QUERY_TAKER = """
query FillsTaker($tokens: [String!]!, $startTs: BigInt!, $endTs: BigInt!,
                 $cursorId: ID!, $size: Int!) {
  rows: orderFilledEvents(
    first: $size,
    orderBy: id, orderDirection: asc,
    where: {
      takerAssetId: "0",
      makerAssetId_in: $tokens,
      timestamp_gte: $startTs, timestamp_lt: $endTs,
      id_gt: $cursorId
    }
  ) {
    id timestamp makerAmountFilled takerAmountFilled
  }
}
"""


def _subgraph_post(
    session: requests.Session, payload: Dict[str, Any], cache_dir: str
) -> Dict[str, Any]:
    key = "sg_" + _sha1(json.dumps(payload, sort_keys=True))
    cached = cache_get(cache_dir, key)
    if cached is not None:
        return cached
    _throttle()
    r = session.post(SUBGRAPH_URL, json=payload, timeout=DEFAULT_TIMEOUT)
    r.raise_for_status()
    data = r.json()
    if "errors" in data:
        # Subgraph errors may be transient; leave them uncached.
        raise RuntimeError(f"subgraph error: {data['errors']}")
    cache_set(cache_dir, key, data)
    return data


def crawl_market_subgraph(
    session: requests.Session,
    *,
    token_ids: List[str],
    window_start: datetime,
    window_end: datetime,
    cache_dir: str,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Any]]:
    """Aggregate Goldsky fills for both outcome tokens; return (daily, stats)."""
    win_start_ts = int(window_start.timestamp())
    win_end_ts = int(window_end.timestamp())

    daily: Dict[str, Dict[str, float]] = {}
    n_trades = 0
    n_pages = 0

    def _paginate(query: str, usdc_field: str, share_field: str) -> int:
        """Aggregate one USDC side using the supplied amount fields."""
        nonlocal n_pages
        cursor_id = ""
        local_n = 0
        while True:
            n_pages += 1
            payload = {
                "query": query,
                "variables": {
                    "tokens": token_ids,
                    "startTs": str(win_start_ts),
                    "endTs": str(win_end_ts),
                    "cursorId": cursor_id,
                    "size": SUBGRAPH_PAGE_SIZE,
                },
            }
            data = _subgraph_post(session, payload, cache_dir)["data"]
            rows = data.get("rows", []) or []
            if not rows:
                break
            for f in rows:
                try:
                    ts = int(f["timestamp"])
                    usdc = int(f[usdc_field]) / TOKEN_DECIMALS
                    shares = int(f[share_field]) / TOKEN_DECIMALS
                except (KeyError, TypeError, ValueError):
                    continue
                if not (win_start_ts <= ts < win_end_ts):
                    continue
                d = _day_str_from_ts(ts)
                slot = daily.setdefault(
                    d, {"trades": 0, "share_volume": 0.0, "notional": 0.0}
                )
                slot["trades"] += 1
                slot["share_volume"] += shares
                slot["notional"] += usdc
                local_n += 1
            cursor_id = rows[-1]["id"]
            if len(rows) < SUBGRAPH_PAGE_SIZE:
                break
        return local_n

    n_trades += _paginate(_SUBGRAPH_QUERY_MAKER, "makerAmountFilled", "takerAmountFilled")
    n_trades += _paginate(_SUBGRAPH_QUERY_TAKER, "takerAmountFilled", "makerAmountFilled")

    stats = {
        "n_trades": n_trades,
        "n_pages": n_pages,
        "truncated": False,
        "oldest_trade": None,
    }
    return daily, stats


def _densify(
    daily: Dict[str, Dict[str, float]], days: List[str]
) -> List[Dict[str, Any]]:
    """Project a {day: {trades, share_volume, notional}} dict onto a fixed list of day strings."""
    out: List[Dict[str, Any]] = []
    for d in days:
        v = daily.get(d, {})
        out.append(
            {
                "date": d,
                "trades": int(v.get("trades", 0)),
                "share_volume": float(round(v.get("share_volume", 0.0), 6)),
                "notional": float(round(v.get("notional", 0.0), 6)),
            }
        )
    return out


def _winner_daily_volume(
    record: Dict[str, Any],
    by_market: Dict[str, List[Dict[str, Any]]],
    days: List[str],
) -> List[Dict[str, Any]]:
    """Return activity on the market leg corresponding to the realized outcome."""
    win_idx = record.get("winner_market_index")
    markets = record.get("markets") or []
    if win_idx is None or not (0 <= int(win_idx) < len(markets)):
        return _densify({}, days)
    winner_mid = str(markets[int(win_idx)].get("market_id"))
    series = by_market.get(winner_mid)
    if series is None:
        return _densify({}, days)
    return series


def _belief_window(record: Dict[str, Any]) -> Tuple[datetime, datetime, List[str]]:
    """Return (start, exclusive_end, UTC_days) for the belief trajectory."""
    daily_index = record.get("daily_index") or []
    if not daily_index:
        s = _parse_iso_dt(record.get("probability_start_date"))
        e = _parse_iso_dt(record.get("probability_end_date"))
        if s is None or e is None:
            raise ValueError("event has no daily_index or probability dates")
        days = []
        cur = s
        while cur <= e:
            days.append(_day_str(cur))
            cur += timedelta(days=1)
        return s, e + timedelta(days=1), days

    # The price collector supplies UTC-midnight dates.
    days = [_parse_iso_dt(d).strftime("%Y-%m-%d") for d in daily_index]
    start_dt = _parse_iso_dt(daily_index[0])
    last_dt = _parse_iso_dt(daily_index[-1])
    end_dt = last_dt + timedelta(days=1)
    return start_dt, end_dt, days


def process_event(
    record: Dict[str, Any],
    session: requests.Session,
    *,
    cache_dir: str,
    method: str,
) -> Dict[str, Any]:
    enriched = dict(record)

    try:
        win_start, win_end, days = _belief_window(record)
    except Exception as e:
        enriched["daily_volume"] = []
        enriched["daily_volume_by_market"] = {}
        enriched["winner_daily_volume"] = []
        enriched["daily_volume_meta"] = {
            "method": method,
            "error": f"window_select_failed:{e}",
            "n_markets": len(record.get("markets") or []),
            "n_markets_resolved": 0,
            "n_markets_truncated": 0,
        }
        enriched["diagnostics"] = {
            "error": "no_window",
            "probability_date_range": None,
            "volume_date_range": None,
        }
        return enriched

    markets = record.get("markets") or []
    by_market: Dict[str, List[Dict[str, Any]]] = {}
    per_market: Dict[str, Dict[str, Any]] = {}
    aggregate: Dict[str, Dict[str, float]] = defaultdict(
        lambda: {"trades": 0, "share_volume": 0.0, "notional": 0.0}
    )
    n_resolved = 0
    n_truncated = 0
    all_trade_days: List[str] = []

    for m in markets:
        mid = str(m.get("market_id"))
        cid = m.get("condition_id")
        token_ids = [t for t in (m.get("clob_token_ids") or []) if t]
        if method in ("subgraph", "auto"):
            ready = bool(token_ids)
        else:
            ready = bool(cid)
        if not ready:
            per_market[mid] = {
                "condition_id": cid,
                "n_trades": 0,
                "n_pages": 0,
                "truncated": False,
                "oldest_trade": None,
                "error": (
                    "missing_clob_token_ids" if method in ("subgraph", "auto")
                    else "missing_condition_id"
                ),
            }
            by_market[mid] = _densify({}, days)
            continue

        n_resolved += 1
        try:
            if method == "subgraph":
                daily, stats = crawl_market_subgraph(
                    session,
                    token_ids=[str(t) for t in token_ids],
                    window_start=win_start,
                    window_end=win_end,
                    cache_dir=cache_dir,
                )
                stats["condition_id"] = cid
            elif method == "rest":
                daily, stats = crawl_market_trades(
                    session,
                    condition_id=cid,
                    window_start=win_start,
                    window_end=win_end,
                    cache_dir=cache_dir,
                )
            else:  # Auto mode falls back to REST when the subgraph fails.
                try:
                    daily, stats = crawl_market_subgraph(
                        session,
                        token_ids=[str(t) for t in token_ids],
                        window_start=win_start,
                        window_end=win_end,
                        cache_dir=cache_dir,
                    )
                    stats["condition_id"] = cid
                    stats["backend"] = "subgraph"
                except Exception:
                    if not cid:
                        raise
                    daily, stats = crawl_market_trades(
                        session,
                        condition_id=cid,
                        window_start=win_start,
                        window_end=win_end,
                        cache_dir=cache_dir,
                    )
                    stats["backend"] = "rest_fallback"
        except Exception as e:
            per_market[mid] = {
                "condition_id": cid,
                "n_trades": 0,
                "n_pages": 0,
                "truncated": True,
                "oldest_trade": None,
                "error": f"exception:{type(e).__name__}:{e}",
            }
            by_market[mid] = _densify({}, days)
            n_truncated += 1
            continue

        by_market[mid] = _densify(daily, days)
        per_market[mid] = stats
        if stats["truncated"]:
            n_truncated += 1

        for d, v in daily.items():
            aggregate[d]["trades"] += v["trades"]
            aggregate[d]["share_volume"] += v.get("share_volume", 0.0)
            aggregate[d]["notional"] += v["notional"]
            if d in days and v["trades"] > 0:
                all_trade_days.append(d)

    enriched["daily_volume_by_market"] = by_market
    enriched["daily_volume"] = _densify(dict(aggregate), days)
    enriched["winner_daily_volume"] = _winner_daily_volume(record, by_market, days)
    enriched["total_volume_metadata"] = record.get("total_volume")

    overlap_full = sum(1 for d in days if d in aggregate and aggregate[d]["trades"] > 0)
    overlap_partial = len(days) - overlap_full
    total_notional = float(sum(v["notional"] for v in aggregate.values()))
    total_share_volume = float(sum(v.get("share_volume", 0.0) for v in aggregate.values()))
    total_trades = int(sum(v["trades"] for v in aggregate.values()))
    meta_total = record.get("total_volume")
    # Compare observed USDC notional with the metadata total.
    ratio_notional = (
        total_notional / float(meta_total)
        if isinstance(meta_total, (int, float)) and meta_total
        else None
    )

    if all_trade_days:
        vol_range = [min(all_trade_days), max(all_trade_days)]
    else:
        vol_range = None

    method_details_map = {
        "subgraph": (
            "Polymarket Goldsky orderbook subgraph (orderFilledEvents). "
            "Filters by clob token id and timestamp range; no offset cap, so "
            "coverage is complete subject to subgraph reindex lag. Each fill "
            "row records both the USDC amount (notional) and the conditional-"
            "token amount (share_volume). MAKER and TAKER queries cover "
            "disjoint rows (one per fill), so n_trades = MAKER + TAKER."
        ),
        "rest": (
            "data-api/trades supports neither time-bound queries nor ordering "
            "control; offset is hard-capped at 3000 (limit<=1000). We page "
            "newest->oldest until we either walk past window_start (complete "
            "coverage) or hit the offset cap (truncated=True). Each row "
            "carries 'size' (shares) and 'price', so share_volume = sum(size) "
            "and notional = sum(size*price)."
        ),
        "auto": (
            "Prefer subgraph (Goldsky orderFilledEvents, complete history); "
            "fall back to data-api/trades only if the subgraph errors. The "
            "per-market 'backend' field records which path was used."
        ),
    }
    enriched["daily_volume_meta"] = {
        "method": method,
        "method_details": method_details_map.get(method, ""),
        "window_start": win_start.isoformat(),
        "window_end": win_end.isoformat(),
        "n_markets": len(markets),
        "n_markets_resolved": n_resolved,
        "n_markets_truncated": n_truncated,
        "per_market": per_market,
    }
    enriched["diagnostics"] = {
        "probability_date_range": [days[0], days[-1]] if days else None,
        "volume_date_range": vol_range,
        "overlap_days_full": overlap_full,
        "overlap_days_partial": overlap_partial,
        "overlap_days_zero": 0,  # Missing activity days are zero-filled.
        "n_belief_days": len(days),
        "n_markets_truncated": n_truncated,
        "total_trades_observed": total_trades,
        "total_share_volume_observed": float(round(total_share_volume, 2)),
        "total_notional_observed": float(round(total_notional, 2)),
        "total_volume_metadata": meta_total,
        "ratio_notional_to_metadata": (
            None if ratio_notional is None else float(round(ratio_notional, 4))
        ),
    }
    return enriched


def _read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                yield json.loads(ln)


def _load_done(path: str) -> set:
    if not os.path.exists(path):
        return set()
    with open(path, "r", encoding="utf-8") as f:
        return {ln.strip() for ln in f if ln.strip()}


def _append_done(path: str, eid: str) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(eid + "\n")


def run(
    *,
    input_path: str,
    output_path: str,
    cache_dir: str,
    rate_limit: float,
    workers: int,
    method: str,
    force: bool,
) -> int:
    global _LIMITER
    _LIMITER = TokenBucket(rps=rate_limit, burst=max(rate_limit * 1.5, 1.0))

    out_dir = os.path.dirname(os.path.abspath(output_path)) or "."
    os.makedirs(out_dir, exist_ok=True)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    done_path = output_path + ".done"
    if force:
        for p in (output_path, done_path):
            if os.path.exists(p):
                os.remove(p)

    done: set = _load_done(done_path)
    total = sum(1 for _ in _read_jsonl(input_path))

    out_mode = "a" if (not force and os.path.exists(output_path)) else "w"
    out_f = open(output_path, out_mode, encoding="utf-8")
    bar = tqdm(total=total, desc="Events", unit="ev")

    session = make_session()
    n_written = 0
    n_skipped_done = 0
    write_lock = threading.Lock()

    def _process(record: Dict[str, Any]) -> Tuple[str, Optional[Dict[str, Any]], str]:
        eid = str(record.get("event_id"))
        try:
            enriched = process_event(
                record, session, cache_dir=cache_dir, method=method
            )
        except Exception as e:
            return eid, None, f"exception:{type(e).__name__}:{e}"
        return eid, enriched, ""

    try:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=workers) as ex:
            pending = []
            for record in _read_jsonl(input_path):
                eid = str(record.get("event_id"))
                if eid in done:
                    n_skipped_done += 1
                    bar.update(1)
                    continue
                pending.append(ex.submit(_process, record))

            for fut in as_completed(pending):
                bar.update(1)
                try:
                    eid, enriched, reason = fut.result()
                except Exception as e:
                    logger.warning("worker crashed: %s", e)
                    continue
                if enriched is None:
                    logger.warning("event %s failed: %s", eid, reason)
                    continue
                with write_lock:
                    out_f.write(json.dumps(enriched, ensure_ascii=False) + "\n")
                    out_f.flush()
                    _append_done(done_path, eid)
                n_written += 1
                bar.set_postfix(
                    wrote=n_written,
                    done_skip=n_skipped_done,
                    trunc=enriched["daily_volume_meta"].get("n_markets_truncated", 0),
                )
    finally:
        out_f.close()
        bar.close()

    logger.info(
        "Wrote %d new records (%d already done) -> %s",
        n_written,
        n_skipped_done,
        output_path,
    )
    return n_written


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Crawl daily trading volume aligned to crowd-belief trajectory."
    )
    p.add_argument("--input", required=True, help="Price-history JSONL from data.collect_prices.")
    p.add_argument("--output", required=True, help="Output JSONL")
    p.add_argument(
        "--cache-dir",
        default=os.getenv("PM_VOL_CACHE_DIR", "pm_volume_cache"),
        help="Directory for trades-page cache. Empty string disables caching.",
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
            f"Per-event worker threads (default {DEFAULT_WORKERS}). The global "
            "rate limit is shared across workers."
        ),
    )
    p.add_argument(
        "--volume-method",
        choices=["auto", "subgraph", "rest"],
        default="auto",
        help=(
            "subgraph: full-history fills; rest: capped REST history; auto: try subgraph, then REST."
        ),
    )
    p.add_argument("--force", action="store_true", help="Recompute everything.")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    return 0 if run(
        input_path=args.input,
        output_path=args.output,
        cache_dir=args.cache_dir,
        rate_limit=args.rate_limit,
        workers=args.workers,
        method=args.volume_method,
        force=args.force,
    ) >= 0 else 1


if __name__ == "__main__":
    sys.exit(main())
