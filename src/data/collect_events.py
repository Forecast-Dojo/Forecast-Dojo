"""Collect resolved Polymarket event metadata."""

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

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm


GAMMA_EVENTS_URL = "https://gamma-api.polymarket.com/events"

DEFAULT_TIMEOUT = 30
DEFAULT_PAGE_LIMIT = 100
DEFAULT_WORKERS = 1
DEFAULT_MAX_RPS = 8.0

# Gamma rejects offsets above 10,000; advance the date cursor sooner.
SAFE_OFFSET_CAP = 9000

YES_PRICE_HI = 0.99
YES_PRICE_LO = 0.01

logger = logging.getLogger("pm_metadata")
logger.setLevel(logging.INFO)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s"))
    logger.addHandler(h)


class TokenBucket:
    """Thread-safe token bucket for rate limiting outbound HTTP."""

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
        total=8,
        backoff_factor=0.8,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=16, pool_maxsize=16)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    s.headers.update({"User-Agent": "PMMetadataCrawler/1.0"})
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


def _parse_json_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if not isinstance(x, str):
        return [x]
    s = x.strip()
    if not s:
        return []
    try:
        v = json.loads(s)
        return v if isinstance(v, list) else [v]
    except Exception:
        return [t.strip() for t in s.split(",") if t.strip()]


def _to_float(x: Any) -> Optional[float]:
    if x is None or x == "":
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _normalize_price(p: float) -> float:
    if p > 1.0 and p <= 100.0:
        p /= 100.0
    return max(0.0, min(1.0, p))


def _truthy(x: Any) -> Optional[bool]:
    if x is None:
        return None
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(x)
    if isinstance(x, str):
        s = x.strip().lower()
        if s in ("true", "1", "yes"):
            return True
        if s in ("false", "0", "no"):
            return False
    return None


def _yes_index(outcomes: List[Any]) -> Optional[int]:
    for i, o in enumerate(outcomes):
        if str(o).strip().lower() == "yes":
            return i
    return None


def _yes_token_id(market: Dict[str, Any], yes_idx: int) -> Optional[str]:
    clob_ids = _parse_json_list(market.get("clobTokenIds"))
    if len(clob_ids) > yes_idx and clob_ids[yes_idx] is not None:
        return str(clob_ids[yes_idx])

    tokens = market.get("tokens")
    outcomes = _parse_json_list(market.get("outcomes"))
    if isinstance(tokens, list) and len(outcomes) > yes_idx:
        target = str(outcomes[yes_idx])
        for t in tokens:
            if (
                isinstance(t, dict)
                and str(t.get("outcome")) == target
                and t.get("token_id") is not None
            ):
                return str(t["token_id"])
    return None


def _resolve_yes(market: Dict[str, Any], yes_idx: int) -> Optional[int]:
    """Read final outcomePrices and return 1 (YES won), 0 (NO won), or None."""
    raw = _parse_json_list(market.get("outcomePrices"))
    if len(raw) <= yes_idx:
        return None
    p = _to_float(raw[yes_idx])
    if p is None:
        return None
    p = _normalize_price(p)
    if p >= YES_PRICE_HI:
        return 1
    if p <= YES_PRICE_LO:
        return 0
    return None


def _normalize_tags(ev: Dict[str, Any]) -> List[str]:
    tags: List[str] = []
    raw = ev.get("tags")
    if isinstance(raw, list):
        for t in raw:
            if isinstance(t, dict):
                tags.append(
                    str(t.get("slug") or t.get("label") or t.get("name") or "").strip()
                )
            else:
                tags.append(str(t).strip())
    return [t for t in tags if t]


def fetch_events_page(
    session: requests.Session,
    *,
    limit: int,
    offset: int,
    end_date_min: Optional[str],
    end_date_max: Optional[str],
    cache_dir: str,
) -> List[Dict[str, Any]]:
    params: Dict[str, Any] = {
        "closed": "true",
        "limit": limit,
        "offset": offset,
        "order": "endDate",
        "ascending": "false",
    }
    if end_date_min is not None:
        params["end_date_min"] = end_date_min
    if end_date_max is not None:
        params["end_date_max"] = end_date_max

    key = "events_" + _sha1(json.dumps(params, sort_keys=True))
    cached = cache_get(cache_dir, key)
    if cached is not None:
        return cached if isinstance(cached, list) else []

    _throttle()
    r = session.get(GAMMA_EVENTS_URL, params=params, timeout=DEFAULT_TIMEOUT)
    if r.status_code == 422:
        # Treat invalid or exhausted cursor windows as empty.
        return []
    r.raise_for_status()
    data = r.json()
    if not isinstance(data, list):
        data = []
    cache_set(cache_dir, key, data)
    return data


def build_market_record(market: Dict[str, Any]) -> Dict[str, Any]:
    outcomes = _parse_json_list(market.get("outcomes"))
    outcome_prices = [_to_float(p) for p in _parse_json_list(market.get("outcomePrices"))]
    clob_ids = [str(c) for c in _parse_json_list(market.get("clobTokenIds"))]

    yes_idx = _yes_index(outcomes)
    yes_token = _yes_token_id(market, yes_idx) if yes_idx is not None else None
    yes_resolved = _resolve_yes(market, yes_idx) if yes_idx is not None else None

    market_volume = _to_float(market.get("volume"))
    if market_volume is None:
        market_volume = _to_float(market.get("volumeNum"))

    return {
        "market_id": str(market.get("id")) if market.get("id") is not None else None,
        "question": (market.get("question") or "").strip() or None,
        "label": (market.get("groupItemTitle") or market.get("question") or "").strip()
        or None,
        "condition_id": market.get("conditionId"),
        "outcomes": [str(o) for o in outcomes],
        "outcome_prices_final": outcome_prices,
        "clob_token_ids": clob_ids,
        "yes_token_id": yes_token,
        "yes_outcome_index": yes_idx,
        "yes_resolved": yes_resolved,
        "closed": _truthy(market.get("closed")),
        "active": _truthy(market.get("active")),
        "archived": _truthy(market.get("archived")),
        "umaResolutionStatus": market.get("umaResolutionStatus"),
        "volume": market_volume,
        "end_date": market.get("endDate") or market.get("endDateIso"),
        "created_at": market.get("createdAt"),
        # Market opening time, distinct from record creation.
        "start_date": market.get("startDate"),
        # Recorded settlement time, distinct from the creator-set deadline.
        "closed_time": market.get("closedTime"),
    }


def derive_ground_truth(
    event: Dict[str, Any], markets: List[Dict[str, Any]]
) -> Tuple[str, Any, Optional[int]]:
    """Return (status, resolved_label, winner_market_index)."""
    neg_risk = bool(event.get("negRisk", False))
    n = len(markets)

    yes_indicators = [m["yes_resolved"] for m in markets]

    if not neg_risk and n == 1:
        v = yes_indicators[0]
        if v == 1:
            return "resolved", "yes", 0
        if v == 0:
            return "resolved", "no", 0
        return "unresolved", None, None

    if neg_risk and n >= 2:
        winners = [i for i, v in enumerate(yes_indicators) if v == 1]
        # Mutually exclusive events require exactly one YES-resolved option.
        if len(winners) == 1:
            idx = winners[0]
            label = markets[idx].get("label") or markets[idx].get("question")
            return "resolved", label, idx
        return "unresolved", None, None

    return "unresolved", None, None


def build_event_record(event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    raw_markets = event.get("markets")
    if not isinstance(raw_markets, list) or not raw_markets:
        return None

    market_records = [build_market_record(m) for m in raw_markets]
    status, label, winner_idx = derive_ground_truth(event, market_records)

    total_volume = _to_float(event.get("volume"))
    if total_volume is None:
        total_volume = _to_float(event.get("volumeNum"))

    return {
        "event_id": str(event.get("id")) if event.get("id") is not None else None,
        "slug": event.get("slug"),
        "title": (event.get("title") or "").strip() or None,
        "body": (event.get("description") or event.get("subtitle") or "").strip()
        or None,
        "creation_date": event.get("createdAt"),
        "start_date": event.get("startDate"),
        "end_date": event.get("endDate"),
        "close_date": event.get("closedTime"),
        "resolution_date": event.get("endDate"),
        "active": _truthy(event.get("active")),
        "closed": _truthy(event.get("closed")),
        "archived": _truthy(event.get("archived")),
        "neg_risk": bool(event.get("negRisk", False)),
        "total_volume": total_volume,
        "tags": _normalize_tags(event),
        "category": event.get("category"),
        "num_markets": len(market_records),
        "markets": market_records,
        "ground_truth_status": status,
        "resolved_label": label,
        "winner_market_index": winner_idx,
    }


def _is_event_resolved(event: Dict[str, Any]) -> bool:
    """Accept closed or archived events."""
    closed = _truthy(event.get("closed"))
    if closed:
        return True
    # Resolved events may be archived without a closed flag.
    if _truthy(event.get("archived")):
        return True
    return False


def _event_end_date(event: Dict[str, Any]) -> Optional[str]:
    return event.get("endDate")


def crawl(
    *,
    start_date: str,
    end_date: str,
    output_path: str,
    page_limit: int,
    cache_dir: str,
    keep_unresolved: bool,
    max_pages: Optional[int],
) -> int:
    """Write resolved events using a descending date cursor; return the record count."""
    session = make_session()

    cur_end_max: Optional[str] = end_date
    seen: set = set()
    pages = 0
    written = 0

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    out = open(output_path, "w", encoding="utf-8")
    out_lock = threading.Lock()

    page_bar = tqdm(desc="Pages", unit="page", position=0)
    ev_bar = tqdm(desc="Events kept", unit="ev", position=1)

    try:
        while True:
            offset = 0
            window_min_end: Optional[str] = None
            window_new = 0

            while offset < SAFE_OFFSET_CAP:
                events = fetch_events_page(
                    session,
                    limit=page_limit,
                    offset=offset,
                    end_date_min=start_date,
                    end_date_max=cur_end_max,
                    cache_dir=cache_dir,
                )
                if not events:
                    break

                for ev in events:
                    end_iso = _event_end_date(ev)
                    if end_iso is not None:
                        if window_min_end is None or end_iso < window_min_end:
                            window_min_end = end_iso

                for ev in events:
                    eid = str(ev.get("id"))
                    if eid in seen:
                        continue
                    seen.add(eid)
                    window_new += 1

                    if not _is_event_resolved(ev):
                        continue

                    record = build_event_record(ev)
                    if record is None:
                        continue

                    if (
                        not keep_unresolved
                        and record["ground_truth_status"] != "resolved"
                    ):
                        continue

                    with out_lock:
                        out.write(json.dumps(record, ensure_ascii=False) + "\n")
                        out.flush()
                    written += 1
                    ev_bar.update(1)

                offset += page_limit
                pages += 1
                page_bar.update(1)
                if max_pages is not None and pages >= max_pages:
                    return written

            if window_min_end is None:
                break

            # Repeated boundary timestamps need an explicit cursor step.
            next_max = window_min_end
            if window_new == 0:
                from datetime import datetime, timedelta, timezone

                try:
                    dt = datetime.fromisoformat(next_max.replace("Z", "+00:00"))
                    dt = dt - timedelta(seconds=1)
                    next_max = (
                        dt.astimezone(timezone.utc)
                        .isoformat()
                        .replace("+00:00", "Z")
                    )
                except Exception:
                    break

            if cur_end_max is not None and next_max >= cur_end_max:
                break

            if next_max <= start_date:
                break

            cur_end_max = next_max
    finally:
        out.close()
        page_bar.close()
        ev_bar.close()

    return written


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Crawl resolved Polymarket event metadata into JSONL."
    )
    p.add_argument("--start-date", required=True, help="YYYY-MM-DD (inclusive)")
    p.add_argument("--end-date", required=True, help="YYYY-MM-DD (inclusive)")
    p.add_argument("--output", required=True, help="Output JSONL path")
    p.add_argument(
        "--page-limit",
        type=int,
        default=DEFAULT_PAGE_LIMIT,
        help=f"Events per Gamma page (default {DEFAULT_PAGE_LIMIT})",
    )
    p.add_argument(
        "--max-rps",
        type=float,
        default=DEFAULT_MAX_RPS,
        help=f"Global cap on outbound HTTP rps (default {DEFAULT_MAX_RPS})",
    )
    p.add_argument(
        "--cache-dir",
        type=str,
        default=os.getenv("PM_META_CACHE_DIR", "pm_meta_cache"),
        help="Directory to cache page responses. Empty string disables caching.",
    )
    p.add_argument(
        "--keep-unresolved",
        action="store_true",
        help="Keep records whose ground truth couldn't be derived (still tagged ground_truth_status=unresolved).",
    )
    p.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Stop after this many pages. Useful for smoke tests.",
    )
    return p.parse_args(argv)


def _normalize_date(s: str, end: bool) -> str:
    """Pass-through if it already looks ISO; else attach time component."""
    if "T" in s:
        return s
    return f"{s}T{'23:59:59Z' if end else '00:00:00Z'}"


def main(argv: Optional[List[str]] = None) -> int:
    global _LIMITER
    args = parse_args(argv)

    if args.start_date > args.end_date:
        logger.error("start-date must be <= end-date")
        return 2

    _LIMITER = TokenBucket(rps=args.max_rps, burst=max(args.max_rps * 1.5, 1.0))

    start_iso = _normalize_date(args.start_date, end=False)
    end_iso = _normalize_date(args.end_date, end=True)

    logger.info(
        "Crawling resolved Polymarket events: end_date in [%s, %s] -> %s",
        start_iso,
        end_iso,
        args.output,
    )

    n = crawl(
        start_date=start_iso,
        end_date=end_iso,
        output_path=args.output,
        page_limit=args.page_limit,
        cache_dir=args.cache_dir,
        keep_unresolved=args.keep_unresolved,
        max_pages=args.max_pages,
    )
    logger.info("Wrote %d event records to %s", n, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
