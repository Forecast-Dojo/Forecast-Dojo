"""Assign article dates from structured metadata and crawl timestamps."""

from __future__ import annotations

import html as _html
import json
import re
from datetime import datetime, timezone


_JSONLD_RE = re.compile(
    r'<script[^>]*type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
    re.DOTALL | re.I)
_OG_PUB_RE = re.compile(
    r'property=["\']article:published_time["\'][^>]*content=["\']([^"\']+)', re.I)
_OG_MOD_RE = re.compile(
    r'property=["\']article:modified_time["\'][^>]*content=["\']([^"\']+)', re.I)
_ITEMPROP_PUB_RE = re.compile(
    r'itemprop=["\']datePublished["\'][^>]*(?:content|datetime)=["\']([^"\']+)', re.I)
_ITEMPROP_MOD_RE = re.compile(
    r'itemprop=["\']dateModified["\'][^>]*(?:content|datetime)=["\']([^"\']+)', re.I)
_TIME_RE = re.compile(r'<time[^>]*datetime=["\']([^"\']+)', re.I)

_ARTICLE_TYPES = {
    "NewsArticle", "Article", "ReportageNewsArticle", "BlogPosting",
    "LiveBlogPosting", "OpinionNewsArticle", "AnalysisNewsArticle",
    "BackgroundNewsArticle", "ReviewNewsArticle",
}


def _try_load(raw: str):
    """Best-effort JSON parse: strict, unescaped, then trailing-comma repair."""
    raw = raw.strip()
    if not raw:
        return None
    for attempt in (raw, _html.unescape(raw)):
        try:
            return json.loads(attempt)
        except Exception:
            pass
    repaired = re.sub(r",(\s*[}\]])", r"\1", _html.unescape(raw))
    try:
        return json.loads(repaired)
    except Exception:
        return None


def _iter_objects(data):
    if isinstance(data, dict):
        yield data
        g = data.get("@graph")
        if isinstance(g, list):
            for x in g:
                yield from _iter_objects(x)
    elif isinstance(data, list):
        for x in data:
            yield from _iter_objects(x)


def _is_article(obj) -> bool:
    t = obj.get("@type")
    types = t if isinstance(t, list) else [t]
    return any(str(x) in _ARTICLE_TYPES or str(x).endswith("Article")
               for x in types if x)


def extract_raw_dates(html: str | bytes):
    """Return publication/modification values and source labels.

    Prefer article JSON-LD, then other JSON-LD, OpenGraph, itemprop, and time."""
    if isinstance(html, bytes):
        html = html.decode("utf-8", "replace")

    article_pub = any_pub = article_mod = any_mod = None
    for m in _JSONLD_RE.finditer(html):
        data = _try_load(m.group(1))
        if data is None:
            continue
        for obj in _iter_objects(data):
            if not isinstance(obj, dict):
                continue
            p = obj.get("datePublished") or obj.get("dateCreated")
            mo = obj.get("dateModified")
            is_art = _is_article(obj)
            if p:
                if is_art and article_pub is None:
                    article_pub = p
                if any_pub is None:
                    any_pub = p
            if mo:
                if is_art and article_mod is None:
                    article_mod = mo
                if any_mod is None:
                    any_mod = mo

    dp = dm = dp_src = dm_src = None
    if article_pub:
        dp, dp_src = article_pub, "jsonld_article"
    elif any_pub:
        dp, dp_src = any_pub, "jsonld_any"
    if article_mod:
        dm, dm_src = article_mod, "jsonld_article"
    elif any_mod:
        dm, dm_src = any_mod, "jsonld_any"

    if dp is None:
        m = _OG_PUB_RE.search(html)
        if m:
            dp, dp_src = m.group(1), "og"
    if dp is None:
        m = _ITEMPROP_PUB_RE.search(html)
        if m:
            dp, dp_src = m.group(1), "itemprop"
    if dp is None:
        m = _TIME_RE.search(html)
        if m:
            dp, dp_src = m.group(1), "time"

    if dm is None:
        m = _OG_MOD_RE.search(html)
        if m:
            dm, dm_src = m.group(1), "og"
    if dm is None:
        m = _ITEMPROP_MOD_RE.search(html)
        if m:
            dm, dm_src = m.group(1), "itemprop"

    return dp, dm, dp_src, dm_src


def parse_dt(s) -> datetime | None:
    """Parse a timezone-aware timestamp, preserving offsets; assume UTC if naive."""
    if s is None:
        return None
    s = str(s).strip()
    if not s:
        return None
    s2 = s.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s2)
    except ValueError:
        head = s[:10]
        try:
            dt = datetime.fromisoformat(head)
        except ValueError:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def to_date_str(dt: datetime | None) -> str | None:
    """Return the UTC day, falling back to the local day on conversion overflow."""
    if dt is None:
        return None
    try:
        return dt.astimezone(timezone.utc).date().isoformat()
    except (OverflowError, ValueError, OSError):
        return dt.date().isoformat()


# Reject structured timestamps outside the publication-year window.
_SANE_YEAR_MIN = 2010
_SANE_YEAR_MAX = 2030


def _is_sane(dt: datetime | None) -> bool:
    """Check the allowed publication-year window."""
    return dt is not None and _SANE_YEAR_MIN <= dt.year <= _SANE_YEAR_MAX


def resolve_published_date(date_published_raw, date_modified_raw, date_crawled_raw):
    """Resolve the closest structured timestamp, capped at crawl time.

    Invalid structured years fall back to the crawl day. Return the resolved
    UTC day, source label, and normalized source dates."""
    dp = parse_dt(date_published_raw)
    dm = parse_dt(date_modified_raw)
    dc = parse_dt(date_crawled_raw)
    dp_n, dm_n, dc_n = to_date_str(dp), to_date_str(dm), to_date_str(dc)

    candidates = []
    if _is_sane(dp):
        candidates.append((dp, "published"))
    if _is_sane(dm):
        candidates.append((dm, "modified"))

    if candidates and dc is not None:
        # Break equal-distance ties toward the later timestamp.
        chosen_dt, label = min(
            candidates,
            key=lambda c: (abs((c[0] - dc).total_seconds()), -c[0].timestamp()),
        )
        clamped = min(chosen_dt, dc)  # Content cannot post-date its crawl.
        published = to_date_str(clamped)
        source = f"closest_{label}"
    elif candidates and dc is None:
        chosen_dt, label = candidates[0]
        published = to_date_str(chosen_dt)
        source = f"no_crawl_{label}"
    elif dc is not None:
        published = dc_n
        source = "crawl_fallback"
    else:
        published = None
        source = "none"

    return {
        "published_date": published,
        "date_source": source,
        "date_published": dp_n,
        "date_modified": dm_n,
        "date_crawled": dc_n,
    }


def extract_published_date(html: str | bytes, date_crawled_raw):
    """Resolve structured HTML dates against the crawl timestamp."""
    dp_raw, dm_raw, dp_src, dm_src = extract_raw_dates(html)
    out = resolve_published_date(dp_raw, dm_raw, date_crawled_raw)
    out["pub_src"] = dp_src
    out["mod_src"] = dm_src
    return out
