# Copyright 2025-2026 Strands RL Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Detect unresolved placeholder labels in market questions."""

from __future__ import annotations

import re

# Reserved slots use short letter/digit codes.
_SLOT_CODE = r"(?:[A-Z]{1,2}|\d{1,2}|[A-Z]\d{1,2})"

# Match full labels such as "Song E by Artist E".
PLACEHOLDER_LABEL_RE = re.compile(
    rf"^\s*[A-Z][a-z]+\s+{_SLOT_CODE}"
    rf"(?:\s+by\s+[A-Z][a-z]+\s+{_SLOT_CODE})?"
    r"\s*[\?.!]*\s*$",
)
PLACEHOLDER_LITERAL_RE = re.compile(r"^\s*placeholder\b.*$", re.IGNORECASE)
# Detect reserved-slot names inside question-shaped labels.
PLACEHOLDER_EMBEDDED_RE = re.compile(
    rf"\b(?:Player|Company|Candidate|Team|Option|App|Person|Party|Song|Movie|"
    rf"Show|Album|Artist|Country|Driver|Constructor|Game|Nominee|Contestant|"
    rf"Entrant|Placeholder)\s+{_SLOT_CODE}\b",
)
# Catch-all labels also require zero lifetime volume.
OTHER_LABEL_RE = re.compile(r"^\s*(?:Other|None(?:\s+of\s+the\s+above)?)\s*$", re.IGNORECASE)


def looks_like_placeholder_label(label: str | None) -> bool:
    """Match reserved-slot label patterns; volume is checked separately."""
    if not label:
        return False
    return bool(
        PLACEHOLDER_LABEL_RE.match(label)
        or PLACEHOLDER_LITERAL_RE.match(label)
        or PLACEHOLDER_EMBEDDED_RE.search(label)
        or OTHER_LABEL_RE.match(label)
    )


def market_lifetime_notional(market: dict, q: dict) -> float:
    """Total notional traded on this leg, summed over its full history."""
    market_id = market.get("market_id")
    if not market_id:
        return 0.0
    days = (q.get("daily_volume_by_market") or {}).get(market_id) or []
    total = 0.0
    for d in days:
        v = d.get("notional") if d else None
        if isinstance(v, (int, float)):
            total += float(v)
    return total


def is_placeholder_market(market: dict, q: dict) -> bool:
    """A leg is a placeholder iff label matches a pattern AND lifetime volume is 0."""
    if not looks_like_placeholder_label(market.get("label")):
        return False
    return market_lifetime_notional(market, q) <= 0.0


def filter_question(q: dict, *, min_real_legs: int = 2) -> dict | None:
    """Remove placeholder legs; reject filtered winners or fewer than min_real_legs."""
    if q.get("belief_kind") != "multi_neg_risk":
        return q

    truth = (q.get("resolved_label") or "").strip()
    markets = q.get("markets") or []

    truth_market = next(
        (m for m in markets if (m.get("label") or "").strip() == truth),
        None,
    )
    if truth_market is not None and is_placeholder_market(truth_market, q):
        return None

    keep = [m for m in markets if not is_placeholder_market(m, q)]
    if len(keep) < min_real_legs:
        return None
    if len(keep) == len(markets):
        return q

    cleaned = dict(q)
    cleaned["markets"] = keep
    cleaned["num_markets"] = len(keep)
    return cleaned
