"""Extract, date, clean, and deduplicate CC-News articles."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse, urlunparse

from tqdm import tqdm


DATA_ROOT = Path(os.environ.get("FORECAST_CORPUS_ROOT", "corpus"))

LANGUAGE_THRESHOLD = 0.5
ENGLISH_CONFIDENCE_THRESHOLD = 0.5

# LSH band geometry sets matching probability; the threshold is logged only.
MINHASH_NGRAM_SIZE = 5
MINHASH_NUM_BUCKETS = 8
MINHASH_HASHES_PER_BUCKET = 16
MINHASH_SIMILARITY_THRESHOLD = 0.88
TITLE_DEDUP_WINDOW_DAYS = 7

SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")
BULLET_LINE_PATTERN = re.compile(r"^\s*(?:[•\-*]|\(?\d+\)?[.)]|[A-Za-z][.)])\s+")

URL_PATTERN = re.compile(
    r"""(?ix)
    \b(
        https?://[^\s<>"'()]+
        |
        www\.[^\s<>"'()]+
    )
    """
)


# Text cleaning


def to_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def normalize_newlines(text: str) -> str:
    """Normalize line endings to LF."""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def remove_urls_preserve_lines(text: str) -> str:
    """Remove inline URLs while preserving line breaks."""
    return URL_PATTERN.sub("", text)


def collapse_inline_spaces(line: str) -> str:
    """Collapse horizontal whitespace and strip the line."""
    line = re.sub(r"[ \t\f\v]+", " ", line)
    return line.strip()


def limit_consecutive_blank_lines(text: str, max_blank_lines: int = 2) -> str:
    """Cap consecutive blank lines and normalize horizontal whitespace."""
    lines = text.split("\n")
    cleaned_lines: list[str] = []
    blank_run = 0

    for raw_line in lines:
        line = raw_line.strip()
        if line == "":
            blank_run += 1
            if blank_run <= max_blank_lines:
                cleaned_lines.append("")
        else:
            blank_run = 0
            cleaned_lines.append(collapse_inline_spaces(raw_line))

    return "\n".join(cleaned_lines).strip()


def clean_extracted_text(text: object, max_blank_lines: int = 2) -> str:
    """Normalize whitespace and remove inline URLs from extracted text."""
    cleaned = to_text(text)
    cleaned = normalize_newlines(cleaned)
    cleaned = remove_urls_preserve_lines(cleaned)
    cleaned = limit_consecutive_blank_lines(cleaned, max_blank_lines=max_blank_lines)
    return cleaned


# Document and language helpers


def normalize_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        text = value
    else:
        text = str(value)
    return " ".join(text.split())


def build_document_text(title: object, summary: object, text: object) -> str:
    parts = [normalize_text(title), normalize_text(summary), normalize_text(text)]
    return "\n\n".join(part for part in parts if part).strip()


def batched(items: list[str], batch_size: int) -> Iterable[list[str]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def clean_label(label: str) -> str:
    prefix = "__label__"
    return label[len(prefix):] if label.startswith(prefix) else label


def truncate_text(text: Any, max_chars: int) -> str:
    value = "" if text is None else str(text)
    if len(value) <= max_chars:
        return value
    return value[:max_chars].rstrip() + "..."


# Standalone cleaning


def _build_step1_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}.step_1.feather")


def _build_step1_compare_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}.step_1.compare.json")


def _build_step1_removed_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}.step_1.removed.json")


def _summarize_step1_examples(df, max_examples: int) -> None:
    changed = df[df["row_changed"]].copy()
    if changed.empty:
        print("No rows changed after cleaning.")
        return

    changed = changed.sort_values("total_char_delta", ascending=False)
    print("=" * 100)
    print(f"Representative examples that changed (top {min(max_examples, len(changed))})")
    print("=" * 100)

    for _, row in changed.head(max_examples).iterrows():
        print(f"Index      : {row['row_index']}")
        print(f"Title      : {row.get('title')}")
        print(f"URL        : {row.get('url')}")
        print(f"Total delta: {row['total_char_delta']}")
        for column in ("title", "summary", "text"):
            original_key = f"original_{column}"
            cleaned_key = f"cleaned_{column}"
            delta_key = f"{column}_char_delta"
            changed_key = f"{column}_changed"
            if not row[changed_key]:
                continue
            before = row[original_key]
            after = row[cleaned_key]
            print(f"{column} before len : {len(before)}")
            print(f"{column} after len  : {len(after)}")
            print(f"{column} char delta : {row[delta_key]}")
            print(f"{column} before     : {before[:300]!r}")
            print(f"{column} after      : {after[:300]!r}")
        print("-" * 100)


def cmd_clean(args: argparse.Namespace) -> None:
    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: pandas. Install it with `pip install pandas pyarrow`."
        ) from exc

    input_path = args.input.expanduser().resolve()
    output_path = (
        args.output.expanduser().resolve()
        if args.output
        else _build_step1_output_path(input_path)
    )
    compare_output_path = (
        args.compare_output.expanduser().resolve()
        if args.compare_output
        else _build_step1_compare_path(input_path)
    )
    removed_output_path = (
        args.removed_output.expanduser().resolve()
        if args.removed_output
        else _build_step1_removed_path(input_path)
    )

    if not input_path.exists():
        raise SystemExit(f"Input feather file not found: {input_path}")

    df = pd.read_feather(input_path)
    target_columns = [c for c in ("title", "summary", "text") if c in df.columns]
    missing_columns = [c for c in ("title", "summary", "text") if c not in df.columns]
    if missing_columns:
        raise SystemExit(
            f"Missing required columns: {missing_columns}. Available columns: {list(df.columns)}"
        )

    df = df.copy()
    df["row_index"] = df.index
    df["total_char_delta"] = 0

    for column in target_columns:
        original_key = f"original_{column}"
        cleaned_key = f"cleaned_{column}"
        changed_key = f"{column}_changed"
        delta_key = f"{column}_char_delta"

        df[original_key] = df[column].fillna("").astype(str)
        df[cleaned_key] = df[original_key].map(
            lambda value: clean_extracted_text(value, max_blank_lines=args.max_blank_lines)
        )
        df[column] = df[cleaned_key]
        df[changed_key] = df[original_key] != df[cleaned_key]
        df[delta_key] = df[original_key].str.len() - df[cleaned_key].str.len()
        df["total_char_delta"] += df[delta_key]

    df["row_changed"] = False
    for column in target_columns:
        df["row_changed"] = df["row_changed"] | df[f"{column}_changed"]

    title_non_empty_mask = df["title"].fillna("").astype(str).str.strip() != ""
    text_non_empty_mask = df["text"].fillna("").astype(str).str.strip() != ""
    required_non_empty_mask = title_non_empty_mask & text_non_empty_mask
    removed_missing_required = int((~required_non_empty_mask).sum())
    removed_df = df[~required_non_empty_mask].copy().reset_index(drop=True)
    kept_df = df[required_non_empty_mask].copy().reset_index(drop=True)

    export_columns_to_drop = ["row_changed", "total_char_delta"]
    for column in target_columns:
        export_columns_to_drop.extend([
            f"original_{column}",
            f"cleaned_{column}",
            f"{column}_changed",
            f"{column}_char_delta",
        ])

    export_df = kept_df.drop(columns=export_columns_to_drop)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_df.to_feather(output_path)

    compare_records = []
    for _, row in kept_df[kept_df["row_changed"]].iterrows():
        record = {
            "row_index": int(row["row_index"]),
            "url": None if row.get("url") is None else str(row.get("url")),
            "total_char_delta": int(row["total_char_delta"]),
        }
        for column in target_columns:
            record[f"{column}_changed"] = bool(row[f"{column}_changed"])
            record[f"{column}_char_delta"] = int(row[f"{column}_char_delta"])
            record[f"original_{column}"] = row[f"original_{column}"]
            record[f"cleaned_{column}"] = row[f"cleaned_{column}"]
        compare_records.append(record)

    compare_output_path.parent.mkdir(parents=True, exist_ok=True)
    compare_output_path.write_text(
        json.dumps(compare_records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    removed_records = []
    for _, row in removed_df.iterrows():
        removed_records.append({
            "row_index": int(row["row_index"]),
            "url": None if row.get("url") is None else str(row.get("url")),
            "removal_step": "step_1",
            "removal_reason": "empty_title_or_text_after_cleaning",
            "original_title": row["original_title"],
            "cleaned_title": row["cleaned_title"],
            "original_summary": row["original_summary"],
            "cleaned_summary": row["cleaned_summary"],
            "original_text": row["original_text"],
            "cleaned_text": row["cleaned_text"],
        })

    removed_output_path.parent.mkdir(parents=True, exist_ok=True)
    removed_output_path.write_text(
        json.dumps(removed_records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Input file            : {input_path}")
    print(f"Output file           : {output_path}")
    print(f"Compare output file   : {compare_output_path}")
    print(f"Removed output file   : {removed_output_path}")
    print(f"Total rows            : {len(df)}")
    print(f"Removed empty title/text: {removed_missing_required}")
    print(f"Kept rows             : {len(kept_df)}")
    print(f"Changed rows kept     : {int(kept_df['row_changed'].sum())}")
    print(f"Unchanged rows kept   : {int((~kept_df['row_changed']).sum())}")
    for column in target_columns:
        print(f"Changed {column:<11}: {int(kept_df[f'{column}_changed'].sum())}")

    _summarize_step1_examples(kept_df, max_examples=args.show_examples)


# Language detection


DEFAULT_LANG_INPUT = DATA_ROOT / "example.feather"
DEFAULT_LANG_FILTERED_INPUT = DATA_ROOT / "example.language_filtered.feather"
DEFAULT_LANG_MODEL = Path(os.environ.get("LID_MODEL_PATH", "models/lid.176.bin"))


def detect_languages(input_path: Path, output_path: Path, model_path: Path, batch_size: int) -> None:
    import fasttext
    import pandas as pd

    if not input_path.exists():
        raise FileNotFoundError(f"Input feather file not found: {input_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"FastText model not found: {model_path}")

    df = pd.read_feather(input_path)

    if "title" not in df.columns:
        raise KeyError(f"'title' column not found in {input_path}")
    if "summary" not in df.columns:
        raise KeyError(f"'summary' column not found in {input_path}")
    if "text" not in df.columns:
        raise KeyError(f"'text' column not found in {input_path}")

    documents = [
        build_document_text(title=row_title, summary=row_summary, text=row_text)
        for row_title, row_summary, row_text in zip(df["title"], df["summary"], df["text"])
    ]

    model = fasttext.load_model(str(model_path))

    languages: list[str] = []
    confidences: list[float] = []
    for batch in batched(documents, batch_size=batch_size):
        safe_batch = [doc if doc else " " for doc in batch]
        safe_batch = [text.replace('\n', ' ') for text in safe_batch]
        labels, scores = model.predict(safe_batch, k=1)
        for label_list, score_list in zip(labels, scores):
            label = clean_label(label_list[0]) if label_list else ""
            score = float(score_list[0]) if score_list else 0.0
            languages.append(label)
            confidences.append(score)

    result = df.copy()
    result["primary_language"] = languages
    result["language_confidence"] = confidences
    filtered_result = result[
        (result["primary_language"] == "en")
        & (result["language_confidence"] >= ENGLISH_CONFIDENCE_THRESHOLD)
    ].reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    filtered_result.to_feather(output_path)

    print(f"Loaded rows            : {len(df):,}")
    print(f"Kept rows              : {len(filtered_result):,}")
    print(f"Input file             : {input_path}")
    print(f"Output file            : {output_path}")
    print(f"Model file             : {model_path}")
    print("Filter                 : primary_language == 'en' and language_confidence >= 0.5")
    print("Added columns          : primary_language, language_confidence")

    filtered_documents = [
        build_document_text(title=row_title, summary=row_summary, text=row_text)
        for row_title, row_summary, row_text in zip(
            filtered_result["title"], filtered_result["summary"], filtered_result["text"]
        )
    ]
    sample_indices = [idx for idx, doc in enumerate(filtered_documents) if doc.strip()][:5]
    if sample_indices:
        print("\nExample model inputs (first 5 non-empty news articles):")
        for sample_index in sample_indices:
            sample_text = filtered_documents[sample_index]
            sample_preview = sample_text[:500]
            if len(sample_text) > 500:
                sample_preview += "..."
            print(f"  row_index            : {sample_index}")
            print(f"  primary_language     : {filtered_result.iloc[sample_index]['primary_language']}")
            print(f"  confidence           : {filtered_result.iloc[sample_index]['language_confidence']:.4f}")
            print("  text_preview         :")
            for line in sample_preview.splitlines() or [""]:
                print(f"    {line}")
            print()

    preview = (
        filtered_result.groupby("primary_language", dropna=False)["language_confidence"]
        .agg(count="size", avg_confidence="mean")
        .sort_values("count", ascending=False)
        .head(10)
        .reset_index()
    )
    if not preview.empty:
        print("\nTop language predictions:")
        for row in preview.itertuples(index=False):
            print(
                f"  {row.primary_language:<8} "
                f"avg_confidence={row.avg_confidence:.4f} "
                f"count={row.count}"
            )


def export_english_json(filtered_input_path: Path, output_path: Path, preview_chars: int) -> None:
    import pandas as pd

    if not filtered_input_path.exists():
        raise FileNotFoundError(f"Filtered feather file not found: {filtered_input_path}")

    df = pd.read_feather(filtered_input_path)

    required_columns = {
        "primary_language",
        "language_confidence",
        "title",
        "summary",
        "text",
    }
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise KeyError(f"Missing required columns in {filtered_input_path}: {missing_text}")

    english_df = (
        df[df["primary_language"] == "en"]
        .copy()
        .sort_values("language_confidence", ascending=False)
        .reset_index(drop=True)
    )

    records = []
    for row in english_df.itertuples(index=False):
        combined_text = build_document_text(
            title=getattr(row, "title", ""),
            summary=getattr(row, "summary", ""),
            text=getattr(row, "text", ""),
        )
        records.append({
            "language_confidence": float(getattr(row, "language_confidence")),
            "text_preview": truncate_text(combined_text, preview_chars),
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"Filtered input file    : {filtered_input_path}")
    print(f"Output json file       : {output_path}")
    print(f"English rows exported  : {len(records):,}")
    print("Sort order             : language_confidence descending")
    print(f"Preview chars          : {preview_chars}")
    if records:
        print("\nFirst 5 exported items:")
        for idx, record in enumerate(records[:5], start=1):
            print(f"  {idx}. confidence={record['language_confidence']:.4f}")
            for line in record["text_preview"].splitlines() or [""]:
                print(f"     {line}")


def cmd_language(args: argparse.Namespace) -> None:
    if args.mode == "detect":
        output_path = args.output or args.input.with_name(
            f"{args.input.stem}.language_filtered.feather"
        )
        detect_languages(
            input_path=args.input,
            output_path=output_path,
            model_path=args.model,
            batch_size=args.batch_size,
        )
        return

    output_path = args.output or args.filtered_input.with_name(
        f"{args.filtered_input.stem}.en_sorted.json"
    )
    export_english_json(
        filtered_input_path=args.filtered_input,
        output_path=output_path,
        preview_chars=args.preview_chars,
    )


# Document-quality pipeline


DEFAULT_PIPELINE_INPUT = DATA_ROOT / "2025-06" / "CC-NEWS-20250601022535-02377.feather"


def build_step_path(input_path: Path, suffix: str) -> Path:
    return input_path.with_name(f"{input_path.stem}.{suffix}")


LID_MODEL_URL = (
    "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
)


def get_datatrove_lid_model_path() -> Path:
    """Find or atomically download the FastText language model.

    Prefer LID_MODEL_PATH, then the datatrove cache."""
    import os
    import tempfile
    import urllib.request

    from huggingface_hub import cached_assets_path

    candidates: list[Path] = []
    override = os.environ.get("LID_MODEL_PATH")
    if override:
        candidates.append(Path(override))
    cache_dir = Path(
        cached_assets_path(library_name="datatrove", namespace="lid", subfolder="ft176")
    )
    candidates.append(cache_dir / "dl.fbaipublicfiles.com_fasttext_supervised-models_lid.176.bin")

    for p in candidates:
        if p.is_file() and p.stat().st_size > 0:
            return p

    # Download to the first writable candidate directory.
    last_err: Exception | None = None
    for final in candidates:
        try:
            final.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            last_err = e
            continue
        if not os.access(final.parent, os.W_OK):
            continue
        tmp = tempfile.NamedTemporaryFile(
            prefix=".lid176.", suffix=".part", dir=final.parent, delete=False
        )
        tmp.close()
        tmp_path = Path(tmp.name)
        try:
            print(f"[filter_news] downloading LID model -> {final}")
            with urllib.request.urlopen(LID_MODEL_URL, timeout=120) as resp, open(tmp_path, "wb") as out:
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    out.write(chunk)
            if tmp_path.stat().st_size < 1_000_000:
                raise RuntimeError(f"LID model download too small: {tmp_path.stat().st_size} bytes")
            try:
                tmp_path.replace(final)
            except OSError:
                tmp_path.unlink(missing_ok=True)
            return final
        except Exception as e:
            tmp_path.unlink(missing_ok=True)
            last_err = e
            continue

    raise RuntimeError(
        f"failed to obtain FastText LID model. Tried: {[str(p) for p in candidates]}. "
        f"Last error: {last_err!r}. Set LID_MODEL_PATH to a writable location."
    )


def split_sentences(text: str) -> list[str]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return []
    parts = SENTENCE_SPLIT_PATTERN.split(normalized)
    return [part.strip() for part in parts if part.strip()]


def get_nonempty_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def first_matches(items: list[str], predicate, *, limit: int, preview_chars: int) -> list[str]:
    matches: list[str] = []
    for item in items:
        if predicate(item):
            matches.append(truncate_text(item, preview_chars))
            if len(matches) >= limit:
                break
    return matches


def estimate_c4_kept_lines(text: str) -> tuple[list[str], int]:
    from datatrove.pipeline.filters.c4_filters import CITATION_REGEX, POLICY_SUBSTRINGS
    from datatrove.utils.text import split_into_sentences

    lines = text.splitlines()
    kept_lines: list[str] = []
    kept_sentence_count = 0

    for line in lines:
        line = line.strip()
        if not line:
            continue
        words = line.split()
        if any(len(word) > 1000 for word in words):
            continue
        line = CITATION_REGEX.sub("", line)
        if len(words) < 3:
            continue
        line_l = line.lower()
        if "lorem ipsum" in line_l:
            return [], 0
        if "javascript" in line_l:
            continue
        if "{" in line:
            return [], 0
        if any(policy in line_l for policy in POLICY_SUBSTRINGS):
            continue
        kept_sentence_count += len(split_into_sentences(line, "en"))
        kept_lines.append(line)

    return kept_lines, kept_sentence_count


def build_reason_evidence(
    *,
    filter_name: str,
    reason: str,
    text: str,
    preview_chars: int,
) -> dict[str, Any]:
    from datatrove.utils.text import split_into_words

    lines = get_nonempty_lines(text)
    sentences = split_sentences(text)
    evidence: dict[str, Any] = {}

    if filter_name == "gopher_repetition_filter":
        if reason in {"dup_para_frac", "dup_para_char_frac"}:
            paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
            duplicates = [p for p, count in Counter(paragraphs).items() if count > 1]
            evidence["trigger_paragraphs"] = [truncate_text(p, preview_chars) for p in duplicates[:3]]
        elif reason in {"dup_line_frac", "dup_line_char_frac"}:
            duplicates = [line for line, count in Counter(lines).items() if count > 1]
            evidence["trigger_lines"] = [truncate_text(line, preview_chars) for line in duplicates[:5]]
        elif reason.startswith("top_") and reason.endswith("_gram"):
            n = int(reason.split("_")[1])
            words = split_into_words(text, "en")
            ngrams = [" ".join(words[i:i + n]) for i in range(0, max(0, len(words) - n + 1))]
            most_common = Counter(ngrams).most_common(3)
            evidence["trigger_ngrams"] = [
                {"ngram": truncate_text(ngram, preview_chars), "count": count}
                for ngram, count in most_common
                if count > 1
            ]
        elif reason.startswith("duplicated_") and reason.endswith("_n_grams"):
            n = int(reason.split("_")[1])
            words = split_into_words(text, "en")
            ngrams = [" ".join(words[i:i + n]) for i in range(0, max(0, len(words) - n + 1))]
            duplicated = [(ngram, count) for ngram, count in Counter(ngrams).most_common(5) if count > 1]
            evidence["trigger_ngrams"] = [
                {"ngram": truncate_text(ngram, preview_chars), "count": count}
                for ngram, count in duplicated
            ]

    elif filter_name == "gopher_quality_filter":
        words = split_into_words(text, "en")
        non_symbol_words = [w for w in words if any(ch.isalnum() for ch in w)]
        evidence["word_count"] = len(non_symbol_words)
        if reason == "gopher_too_many_hashes":
            evidence["trigger_lines"] = first_matches(
                lines, lambda line: "#" in line, limit=5, preview_chars=preview_chars
            )
        elif reason == "gopher_too_many_ellipsis":
            evidence["trigger_sentences"] = first_matches(
                sentences, lambda s: "..." in s or "…" in s, limit=5, preview_chars=preview_chars
            )
        elif reason == "gopher_too_many_bullets":
            evidence["trigger_lines"] = first_matches(
                lines, lambda line: bool(BULLET_LINE_PATTERN.match(line)), limit=5, preview_chars=preview_chars
            )
        elif reason == "gopher_too_many_end_ellipsis":
            evidence["trigger_lines"] = first_matches(
                lines, lambda line: line.endswith("...") or line.endswith("…"), limit=5, preview_chars=preview_chars
            )
        elif reason == "gopher_below_alpha_threshold":
            evidence["non_alpha_word_samples"] = [
                truncate_text(word, preview_chars) for word in words if not any(ch.isalpha() for ch in word)
            ][:10]
        elif reason == "gopher_enough_stop_words":
            stop_words = {"the", "be", "to", "of", "and", "that", "have", "with"}
            lowered = {w.lower() for w in words}
            evidence["stopword_hits"] = sorted(stop_words.intersection(lowered))

    elif filter_name == "c4_quality_filter":
        if reason == "too_few_sentences":
            kept_lines, kept_sentence_count = estimate_c4_kept_lines(text)
            evidence["kept_sentence_count_after_c4_line_cleaning"] = kept_sentence_count
            evidence["min_required_sentences"] = 5
            evidence["remaining_lines_preview"] = [
                truncate_text(line, preview_chars) for line in kept_lines[:5]
            ]
        elif reason == "lorem_ipsum":
            evidence["trigger_lines"] = first_matches(
                lines, lambda line: "lorem ipsum" in line.lower(), limit=3, preview_chars=preview_chars
            )
        elif reason == "curly_bracket":
            evidence["trigger_lines"] = first_matches(
                lines, lambda line: "{" in line, limit=3, preview_chars=preview_chars
            )

    elif filter_name == "fineweb_quality_filter":
        if reason == "line_punct_ratio":
            evidence["trigger_lines"] = first_matches(
                lines,
                lambda line: not line.endswith((".", "!", "?", '"', "'")),
                limit=5,
                preview_chars=preview_chars,
            )
        elif reason == "short_line_ratio":
            evidence["trigger_lines"] = first_matches(
                lines, lambda line: len(line) <= 30, limit=8, preview_chars=preview_chars
            )
        elif reason == "char_dup_ratio":
            duplicates = [line for line, count in Counter(lines).items() if count > 1]
            evidence["trigger_lines"] = [truncate_text(line, preview_chars) for line in duplicates[:5]]
        elif reason == "list_ratio":
            evidence["trigger_lines"] = first_matches(
                lines, lambda line: bool(BULLET_LINE_PATTERN.match(line)), limit=8, preview_chars=preview_chars
            )
        elif reason == "empty":
            evidence["trigger_lines"] = []

    return evidence


def ensure_row_index(df):
    import pandas as pd
    result = df.copy()
    if "row_index" not in result.columns:
        result["row_index"] = pd.RangeIndex(start=0, stop=len(result), step=1)
    return result


def save_removed_json(
    path: Path,
    *,
    step_name: str,
    filter_name: str,
    input_file: Path,
    output_file: Path,
    removed_records: list[dict[str, Any]],
    kept_count: int,
) -> None:
    reason_counts = Counter(
        str(record.get("removal_reason", "unknown")) for record in removed_records
    )
    payload = {
        "step": step_name,
        "filter": filter_name,
        "input_file": str(input_file),
        "output_file": str(output_file),
        "removed_count": len(removed_records),
        "kept_count": kept_count,
        "reason_counts": dict(sorted(reason_counts.items(), key=lambda item: item[0])),
        "removed_records": removed_records,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def format_reason_counts(reason_counts: dict[str, int]) -> str:
    if not reason_counts:
        return "none"
    parts = [f"{reason}={count}" for reason, count in sorted(reason_counts.items())]
    return ", ".join(parts)


def load_removed_payload(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_date_dir(date_arg: str) -> tuple[Path, str | None, str]:
    """Return (directory, optional filename prefix, aggregate label) for --date."""
    candidate = Path(date_arg).expanduser()
    if candidate.exists():
        if not candidate.is_dir():
            raise SystemExit(f"--date path is not a directory: {candidate}")
        resolved = candidate.resolve()
        return resolved, None, resolved.name

    month_match = re.fullmatch(r"(\d{4})-?(\d{2})", date_arg)
    day_match = re.fullmatch(r"(\d{4})(\d{2})(\d{2})", date_arg)

    if day_match:
        year, month, _day = day_match.groups()
        month_folder = f"{year}-{month}"
        resolved = (DATA_ROOT / month_folder).resolve()
        prefix = f"CC-NEWS-{date_arg}"
        label = date_arg
    elif month_match:
        year, month = month_match.groups()
        month_folder = f"{year}-{month}"
        resolved = (DATA_ROOT / month_folder).resolve()
        prefix = None
        label = month_folder
    else:
        raise SystemExit(
            f"Invalid --date value: {date_arg}. "
            "Use YYYY-MM, YYYYMM, YYYYMMDD, or a directory path."
        )

    if not resolved.exists():
        raise SystemExit(f"Date directory not found: {resolved}")
    return resolved, prefix, label


def is_source_feather(path: Path) -> bool:
    if path.suffix != ".feather":
        return False
    name = path.name
    exclude_markers = (
        ".step_",
        ".language_filtered",
        ".gopher_",
        ".c4_",
        ".fineweb_",
    )
    return not any(marker in name for marker in exclude_markers)


def get_filter_specs(args: argparse.Namespace):
    from datatrove.pipeline.filters import (
        C4QualityFilter,
        FineWebQualityFilter,
        GopherQualityFilter,
        GopherRepetitionFilter,
    )

    if args.fast_order:
        return [
            (
                "step_3",
                "step_3.gopher_quality_filtered.feather",
                "step_3.gopher_quality_removed.json",
                "gopher_quality_filter",
                GopherQualityFilter(
                    max_non_alpha_words_ratio=0.75,
                ),
            ),
            (
                "step_4",
                "step_4.gopher_repetition_filtered.feather",
                "step_4.gopher_repetition_removed.json",
                "gopher_repetition_filter",
                GopherRepetitionFilter(),
            ),
            (
                "step_5",
                "step_5.c4_filtered.feather",
                "step_5.c4_removed.json",
                "c4_quality_filter",
                C4QualityFilter(filter_no_terminal_punct=False),
            ),
            (
                "step_6",
                "step_6.fineweb_quality_filtered.feather",
                "step_6.fineweb_quality_removed.json",
                "fineweb_quality_filter",
                FineWebQualityFilter(char_duplicates_ratio=1.0, new_line_ratio=1.0, short_line_thr=0.75),
            ),
        ]
    return [
        (
            "step_3",
            "step_3.gopher_repetition_filtered.feather",
            "step_3.gopher_repetition_removed.json",
            "gopher_repetition_filter",
            GopherRepetitionFilter(),
        ),
        (
            "step_4",
            "step_4.gopher_quality_filtered.feather",
            "step_4.gopher_quality_removed.json",
            "gopher_quality_filter",
            GopherQualityFilter(
                max_non_alpha_words_ratio=0.75,
            ),
        ),
        (
            "step_5",
            "step_5.c4_filtered.feather",
            "step_5.c4_removed.json",
            "c4_quality_filter",
            C4QualityFilter(filter_no_terminal_punct=False),
        ),
        (
            "step_6",
            "step_6.fineweb_quality_filtered.feather",
            "step_6.fineweb_quality_removed.json",
            "fineweb_quality_filter",
            FineWebQualityFilter(char_duplicates_ratio=1.0, new_line_ratio=1.0, short_line_thr=0.80),
        ),
    ]


def run_step_1(input_path: Path, preview_chars: int):
    import pandas as pd

    df = pd.read_feather(input_path)
    df = ensure_row_index(df)

    target_columns = ("title", "summary", "text")
    missing_columns = [column for column in target_columns if column not in df.columns]
    if missing_columns:
        raise SystemExit(
            f"Missing required columns for step_1: {missing_columns}. "
            f"Available columns: {list(df.columns)}"
        )

    working_df = df.copy()
    working_df["total_char_delta"] = 0

    with tqdm(total=len(target_columns), desc="step_1 clean", unit="field") as pbar:
        for column in target_columns:
            original_key = f"original_{column}"
            cleaned_key = f"cleaned_{column}"
            changed_key = f"{column}_changed"
            delta_key = f"{column}_char_delta"

            working_df[original_key] = working_df[column].fillna("").astype(str)
            working_df[cleaned_key] = working_df[original_key].map(clean_extracted_text)
            working_df[column] = working_df[cleaned_key]
            working_df[changed_key] = working_df[original_key] != working_df[cleaned_key]
            working_df[delta_key] = (
                working_df[original_key].str.len() - working_df[cleaned_key].str.len()
            )
            working_df["total_char_delta"] += working_df[delta_key]
            pbar.update(1)

    working_df["row_changed"] = False
    for column in target_columns:
        working_df["row_changed"] = working_df["row_changed"] | working_df[f"{column}_changed"]

    title_non_empty_mask = working_df["title"].fillna("").astype(str).str.strip() != ""
    text_non_empty_mask = working_df["text"].fillna("").astype(str).str.strip() != ""
    required_non_empty_mask = title_non_empty_mask & text_non_empty_mask
    removed_df = working_df[~required_non_empty_mask].copy().reset_index(drop=True)
    kept_df = working_df[required_non_empty_mask].copy().reset_index(drop=True)

    output_path = build_step_path(input_path, "step_1.feather")
    compare_output_path = build_step_path(input_path, "step_1.compare.json")
    removed_output_path = build_step_path(input_path, "step_1.removed.json")

    export_columns_to_drop = ["row_changed", "total_char_delta"]
    for column in target_columns:
        export_columns_to_drop.extend([
            f"original_{column}",
            f"cleaned_{column}",
            f"{column}_changed",
            f"{column}_char_delta",
        ])

    export_df = kept_df.drop(columns=export_columns_to_drop)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_df.to_feather(output_path)

    compare_records = []
    for _, row in tqdm(
        kept_df[kept_df["row_changed"]].iterrows(),
        total=int(kept_df["row_changed"].sum()),
        desc="step_1 compare json",
        unit="row",
    ):
        record = {
            "row_index": int(row["row_index"]),
            "url": None if row.get("url") is None else str(row.get("url")),
            "total_char_delta": int(row["total_char_delta"]),
        }
        for column in target_columns:
            record[f"{column}_changed"] = bool(row[f"{column}_changed"])
            record[f"{column}_char_delta"] = int(row[f"{column}_char_delta"])
            record[f"original_{column}"] = row[f"original_{column}"]
            record[f"cleaned_{column}"] = row[f"cleaned_{column}"]
        compare_records.append(record)

    compare_output_path.write_text(
        json.dumps(compare_records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    removed_records = []
    for _, row in tqdm(
        removed_df.iterrows(),
        total=len(removed_df),
        desc="step_1 removed json",
        unit="row",
    ):
        removed_records.append({
            "row_index": int(row["row_index"]),
            "url": None if row.get("url") is None else str(row.get("url")),
            "removal_step": "step_1",
            "removal_reason": "empty_title_or_text_after_cleaning",
            "title_preview": truncate_text(row["cleaned_title"], preview_chars),
            "summary_preview": truncate_text(row["cleaned_summary"], preview_chars),
            "text_preview": truncate_text(row["cleaned_text"], preview_chars),
        })

    save_removed_json(
        removed_output_path,
        step_name="step_1",
        filter_name="local_preclean_and_nonempty_title_text",
        input_file=input_path,
        output_file=output_path,
        removed_records=removed_records,
        kept_count=len(export_df),
    )

    summary = {
        "output_path": output_path,
        "compare_output_path": compare_output_path,
        "removed_output_path": removed_output_path,
        "total_rows": len(df),
        "kept_rows": len(export_df),
        "removed_rows": len(removed_records),
    }
    return export_df, summary


def run_language_filter(
    df,
    *,
    input_path: Path,
    language_threshold: float,
    preview_chars: int,
):
    from datatrove.data import Document
    from fasttext.FastText import _FastText

    model_path = get_datatrove_lid_model_path()
    model = _FastText(str(model_path))

    kept_rows: list[dict[str, Any]] = []
    removed_records: list[dict[str, Any]] = []

    records = df.to_dict("records")
    with tqdm(total=len(records), desc="step_2 language", unit="row") as pbar:
        for row in records:
            combined_text = build_document_text(row.get("title"), row.get("summary"), row.get("text"))
            doc = Document(
                text=combined_text,
                id=str(row.get("id", row.get("row_index", ""))),
                metadata={},
            )
            labels, scores = model.predict(doc.text.replace("\n", " "), k=1)
            label = labels[0].split("__")[2] if labels else ""
            score = float(scores[0]) if len(scores) else 0.0
            keep = label == "en" and score >= language_threshold
            if keep:
                kept_rows.append(row)
            else:
                removed_records.append({
                    "row_index": int(row["row_index"]),
                    "url": None if row.get("url") is None else str(row.get("url")),
                    "removal_step": "step_2",
                    "removal_reason": "language_filter",
                    "predicted_language": label,
                    "language_confidence": score,
                    "language_threshold": language_threshold,
                    "title_preview": truncate_text(row.get("title"), preview_chars),
                    "summary_preview": truncate_text(row.get("summary"), preview_chars),
                    "text_preview": truncate_text(row.get("text"), preview_chars),
                    "combined_preview": truncate_text(combined_text, preview_chars),
                })
            pbar.update(1)
            pbar.set_postfix(kept=len(kept_rows), removed=len(removed_records), refresh=False)

    kept_df = df.iloc[0:0].copy() if not kept_rows else type(df)(kept_rows)
    kept_df = kept_df.reset_index(drop=True)

    output_path = build_step_path(input_path, "step_2.language_filtered.feather")
    removed_output_path = build_step_path(input_path, "step_2.language_removed.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    kept_df.to_feather(output_path)
    save_removed_json(
        removed_output_path,
        step_name="step_2",
        filter_name="language_filter",
        input_file=input_path,
        output_file=output_path,
        removed_records=removed_records,
        kept_count=len(kept_df),
    )

    summary = {
        "output_path": output_path,
        "removed_output_path": removed_output_path,
        "kept_rows": len(kept_df),
        "removed_rows": len(removed_records),
    }
    return kept_df, summary


def parse_filter_result(result: Any) -> tuple[bool, str]:
    if isinstance(result, tuple):
        keep, reason = result
        return bool(keep), str(reason)
    return bool(result), "filtered"


def run_datatrove_filter(
    df,
    *,
    input_path: Path,
    step_name: str,
    feather_suffix: str,
    removed_suffix: str,
    filter_name: str,
    filter_instance,
    preview_chars: int,
):
    import pandas as pd
    from datatrove.data import Document

    kept_rows: list[dict[str, Any]] = []
    removed_records: list[dict[str, Any]] = []

    records = df.to_dict("records")
    with tqdm(total=len(records), desc=step_name, unit="row") as pbar:
        for row in records:
            current_text = "" if row.get("text") is None else str(row.get("text"))
            doc = Document(
                text=current_text,
                id=str(row.get("id", row.get("row_index", ""))),
                metadata={"url": row.get("url")},
            )
            keep, reason = parse_filter_result(filter_instance.filter(doc))

            if keep:
                kept_row = row.copy()
                kept_row["text"] = doc.text
                kept_rows.append(kept_row)
            else:
                removed_record = {
                    "row_index": int(row["row_index"]),
                    "url": None if row.get("url") is None else str(row.get("url")),
                    "removal_step": step_name,
                    "removal_reason": reason,
                    "filter_name": filter_name,
                    "title_preview": truncate_text(row.get("title"), preview_chars),
                    "summary_preview": truncate_text(row.get("summary"), preview_chars),
                    "text_preview": truncate_text(current_text, preview_chars),
                }
                if doc.text != current_text:
                    removed_record["post_filter_text_preview"] = truncate_text(doc.text, preview_chars)
                removed_record.update(
                    build_reason_evidence(
                        filter_name=filter_name,
                        reason=reason,
                        text=current_text,
                        preview_chars=preview_chars,
                    )
                )
                removed_records.append(removed_record)
            pbar.update(1)
            pbar.set_postfix(kept=len(kept_rows), removed=len(removed_records), refresh=False)

    kept_df = pd.DataFrame(kept_rows, columns=df.columns).reset_index(drop=True)
    output_path = build_step_path(input_path, feather_suffix)
    removed_output_path = build_step_path(input_path, removed_suffix)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    kept_df.to_feather(output_path)
    save_removed_json(
        removed_output_path,
        step_name=step_name,
        filter_name=filter_name,
        input_file=input_path,
        output_file=output_path,
        removed_records=removed_records,
        kept_count=len(kept_df),
    )

    summary = {
        "output_path": output_path,
        "removed_output_path": removed_output_path,
        "kept_rows": len(kept_df),
        "removed_rows": len(removed_records),
    }
    return kept_df, summary


def run_pipeline_for_file(input_path: Path, args: argparse.Namespace) -> dict[str, Any]:
    print("=" * 100)
    print(f"Processing file: {input_path.name}")
    print("=" * 100)

    step_1_df, step_1_summary = run_step_1(input_path, preview_chars=args.preview_chars)
    source_rows = int(step_1_summary["total_rows"])
    print(f"Step 1 kept/removed   : {step_1_summary['kept_rows']}/{step_1_summary['removed_rows']}")

    step_2_df, step_2_summary = run_language_filter(
        step_1_df,
        input_path=input_path,
        language_threshold=args.language_threshold,
        preview_chars=args.preview_chars,
    )
    print(f"Step 2 kept/removed   : {step_2_summary['kept_rows']}/{step_2_summary['removed_rows']}")

    filter_specs = get_filter_specs(args)
    current_df = step_2_df
    step_summaries: dict[str, dict[str, Any]] = {
        "step_1": step_1_summary,
        "step_2": step_2_summary,
    }

    for step_name, feather_suffix, removed_suffix, filter_name, filter_instance in filter_specs:
        current_df, summary = run_datatrove_filter(
            current_df,
            input_path=input_path,
            step_name=step_name,
            feather_suffix=feather_suffix,
            removed_suffix=removed_suffix,
            filter_name=filter_name,
            filter_instance=filter_instance,
            preview_chars=args.preview_chars,
        )
        step_summaries[step_name] = summary
        print(f"{step_name} kept/removed : {summary['kept_rows']}/{summary['removed_rows']}")

    final_output = build_step_path(input_path, "step_6.fineweb_quality_filtered.feather")
    final_rows = int(len(current_df))
    total_removed = source_rows - final_rows
    print(f"Final output          : {final_output}")
    print(f"Total/Input rows      : {source_rows}")
    print(f"Total/Kept rows       : {final_rows}")
    print(f"Total/Removed rows    : {total_removed}")
    return {
        "input_path": input_path,
        "final_df": current_df,
        "source_rows": source_rows,
        "final_rows": final_rows,
        "total_removed": total_removed,
        "step_summaries": step_summaries,
    }


def _pipeline_worker(args_tuple: tuple[Path, argparse.Namespace]) -> dict[str, Any]:
    """Run one file and return a serializable summary of saved outputs."""
    input_path, args = args_tuple
    try:
        result = run_pipeline_for_file(input_path, args)
    except Exception as exc:
        import traceback
        return {
            "status": "error",
            "input_path": str(input_path),
            "error": f"{exc!r}\n{traceback.format_exc()}",
        }

    step6_path = build_step_path(input_path, "step_6.fineweb_quality_filtered.feather")
    serializable_summaries: dict[str, dict[str, Any]] = {}
    for step_name, summary in result["step_summaries"].items():
        serializable_summaries[step_name] = {
            k: (str(v) if isinstance(v, Path) else v) for k, v in summary.items()
        }
    return {
        "status": "ok",
        "input_path": str(input_path),
        "step6_path": str(step6_path),
        "source_rows": int(result["source_rows"]),
        "final_rows": int(result["final_rows"]),
        "total_removed": int(result["total_removed"]),
        "step_summaries": serializable_summaries,
    }


def build_aggregate_path(date_dir: Path, suffix: str, label: str | None = None) -> Path:
    name = label if label else date_dir.name
    return date_dir / f"{name}.aggregate.{suffix}"


def save_aggregate_reason_summary_json(
    date_dir: Path,
    *,
    reason_counts_by_step: dict[str, dict[str, int]],
    source_file_count: int,
    input_rows_total: int,
    kept_rows_total: int,
    removed_rows_total: int,
    label: str | None = None,
) -> Path:
    output_path = build_aggregate_path(date_dir, "reason_summary.json", label=label)
    payload = {
        "date_dir": str(date_dir),
        "source_file_count": int(source_file_count),
        "total_input_rows": int(input_rows_total),
        "total_kept_rows": int(kept_rows_total),
        "total_removed_rows": int(removed_rows_total),
        "reason_counts_by_step": {
            step: dict(sorted(counts.items(), key=lambda item: item[0]))
            for step, counts in sorted(reason_counts_by_step.items(), key=lambda item: item[0])
        },
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def save_aggregated_removed_json(
    date_dir: Path,
    *,
    step_name: str,
    source_results: list[dict[str, Any]],
    label: str | None = None,
) -> Path:
    removed_records: list[dict[str, Any]] = []
    total_kept = 0
    for item in source_results:
        source_file = item["input_path"]
        summary = item["step_summaries"][step_name]
        removed_path = Path(summary["removed_output_path"])
        payload = load_removed_payload(removed_path)
        total_kept += int(payload.get("kept_count", 0))
        for record in payload.get("removed_records", []):
            merged = dict(record)
            merged["source_file"] = str(source_file)
            removed_records.append(merged)

    output_path = build_aggregate_path(date_dir, f"{step_name}.removed.json", label=label)
    reason_counts = Counter(str(record.get("removal_reason", "unknown")) for record in removed_records)
    aggregate_payload = {
        "date_dir": str(date_dir),
        "step": step_name,
        "removed_count": len(removed_records),
        "kept_count_total": total_kept,
        "reason_counts": dict(sorted(reason_counts.items(), key=lambda item: item[0])),
        "removed_records": removed_records,
    }
    output_path.write_text(json.dumps(aggregate_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def cmd_pipeline(args: argparse.Namespace) -> None:
    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: pandas. Install it with `pip install pandas pyarrow`."
        ) from exc

    if args.date:
        date_dir, filename_prefix, aggregate_label = resolve_date_dir(args.date)
        source_files = sorted(
            path
            for path in date_dir.glob("*.feather")
            if is_source_feather(path)
            and (filename_prefix is None or path.name.startswith(filename_prefix))
        )
        if not source_files:
            hint = f" (prefix filter: {filename_prefix})" if filename_prefix else ""
            raise SystemExit(
                f"No source feather files found in date directory: {date_dir}{hint}"
            )

        print("=" * 100)
        print(f"Batch mode date dir   : {date_dir}")
        if filename_prefix:
            print(f"Filename prefix filter: {filename_prefix}")
        print(f"Aggregate label       : {aggregate_label}")
        print(f"Source files          : {len(source_files)}")
        print("=" * 100)

        from concurrent.futures import ProcessPoolExecutor, as_completed

        workers = max(1, int(getattr(args, "workers", 1) or 1))
        workers = min(workers, len(source_files))
        print(f"Workers               : {workers}")
        print("=" * 100)

        results: list[dict[str, Any]] = []
        if workers == 1:
            for source_file in tqdm(source_files, desc="files", unit="file"):
                full_result = run_pipeline_for_file(source_file.resolve(), args)
                # Match the parallel result format; reload the saved frame below.
                step6_path = build_step_path(source_file.resolve(), "step_6.fineweb_quality_filtered.feather")
                results.append({
                    "status": "ok",
                    "input_path": str(source_file.resolve()),
                    "step6_path": str(step6_path),
                    "source_rows": int(full_result["source_rows"]),
                    "final_rows": int(full_result["final_rows"]),
                    "total_removed": int(full_result["total_removed"]),
                    "step_summaries": full_result["step_summaries"],
                })
        else:
            tasks = [(p.resolve(), args) for p in source_files]
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(_pipeline_worker, t): t[0] for t in tasks}
                with tqdm(total=len(futures), desc="files", unit="file") as pbar:
                    for future in as_completed(futures):
                        src = futures[future]
                        try:
                            res = future.result()
                        except Exception as exc:
                            tqdm.write(f"  ! {src.name}: worker died ({exc!r})")
                            pbar.update(1)
                            continue
                        if res.get("status") == "error":
                            tqdm.write(f"  ! {src.name}: pipeline failed\n{res.get('error', '')}")
                        else:
                            tqdm.write(
                                f"  ✓ {src.name}: kept {res['final_rows']}/{res['source_rows']}"
                            )
                            results.append(res)
                        pbar.update(1)

        if not results:
            raise SystemExit("All source files failed; nothing to aggregate.")

        # Reload saved frames rather than passing them between processes.
        step6_paths = [Path(item["step6_path"]) for item in results]
        final_dfs = [pd.read_feather(p) for p in step6_paths if p.exists()]
        aggregate_final_df = pd.concat(final_dfs, ignore_index=True) if final_dfs else pd.DataFrame()
        aggregate_step6_feather_path = build_aggregate_path(
            date_dir, "step_6.fineweb_quality_filtered.feather", label=aggregate_label
        )
        aggregate_final_df.to_feather(aggregate_step6_feather_path)
        aggregate_source_rows = sum(int(item["source_rows"]) for item in results)
        aggregate_step6_kept_rows = int(len(aggregate_final_df))

        print("=" * 100)
        print(f"Step 7 dedup on aggregate ({aggregate_step6_kept_rows} rows)")
        print("=" * 100)
        dedup_feather_path = build_aggregate_path(
            date_dir, "step_7.dedup_filtered.feather", label=aggregate_label
        )
        dedup_removed_path = build_aggregate_path(
            date_dir, "step_7.dedup_removed.json", label=aggregate_label
        )
        dedup_summary = run_dedup_step(
            aggregate_final_df,
            output_feather_path=dedup_feather_path,
            removed_json_path=dedup_removed_path,
            input_path_for_log=aggregate_step6_feather_path,
            use_url_dedup=not args.no_url_dedup,
            use_title_dedup=not args.no_title_dedup,
            title_window_days=args.title_window_days,
            minhash_ngram_size=args.dedup_ngram_size,
            minhash_num_buckets=args.dedup_num_buckets,
            minhash_hashes_per_bucket=args.dedup_hashes_per_bucket,
            minhash_similarity_threshold=args.dedup_similarity,
            preview_chars=args.preview_chars,
        )
        aggregate_feather_path = dedup_feather_path
        aggregate_kept_rows = int(dedup_summary["kept_rows"])
        print(f"Step 7 kept/removed   : {dedup_summary['kept_rows']}/{dedup_summary['removed_rows']}")

        aggregate_removed_rows = aggregate_source_rows - aggregate_kept_rows

        step_names = ["step_1", "step_2", "step_3", "step_4", "step_5", "step_6"]
        aggregate_json_paths = []
        aggregate_reason_counts: dict[str, dict[str, int]] = {}
        for step_name in step_names:
            path = save_aggregated_removed_json(
                date_dir,
                step_name=step_name,
                source_results=results,
                label=aggregate_label,
            )
            aggregate_json_paths.append(path)
            aggregate_payload = load_removed_payload(path)
            aggregate_reason_counts[step_name] = {
                str(k): int(v) for k, v in aggregate_payload.get("reason_counts", {}).items()
            }

        step_names.append("step_7")
        dedup_payload = load_removed_payload(dedup_removed_path)
        aggregate_reason_counts["step_7"] = {
            str(k): int(v) for k, v in dedup_payload.get("reason_counts", {}).items()
        }
        aggregate_json_paths.append(dedup_removed_path)

        aggregate_reason_summary_path = save_aggregate_reason_summary_json(
            date_dir,
            reason_counts_by_step=aggregate_reason_counts,
            source_file_count=len(source_files),
            input_rows_total=aggregate_source_rows,
            kept_rows_total=aggregate_kept_rows,
            removed_rows_total=aggregate_removed_rows,
            label=aggregate_label,
        )

        print("\n" + "=" * 100)
        print("Batch cleaning completed")
        print("=" * 100)
        print(f"Date dir              : {date_dir}")
        print(f"Aggregate feather     : {aggregate_feather_path}")
        print(f"Total/Input rows      : {aggregate_source_rows}")
        print(f"Total/Kept rows       : {aggregate_kept_rows}")
        print(f"Total/Removed rows    : {aggregate_removed_rows}")
        print("Aggregate removed json:")
        for path in aggregate_json_paths:
            print(f"  - {path}")
        print(f"Aggregate reason summary: {aggregate_reason_summary_path}")
        print("Reason counts by step:")
        for step_name in step_names:
            print(f"  {step_name}: {format_reason_counts(aggregate_reason_counts.get(step_name, {}))}")
        return

    input_path = args.input.expanduser().resolve()
    if not input_path.exists():
        raise SystemExit(f"Input feather file not found: {input_path}")

    result = run_pipeline_for_file(input_path, args)
    step_names = ["step_1", "step_2", "step_3", "step_4", "step_5", "step_6"]
    per_step_reason_counts: dict[str, dict[str, int]] = {}
    for step_name in step_names:
        removed_path = Path(result["step_summaries"][step_name]["removed_output_path"])
        payload = load_removed_payload(removed_path)
        per_step_reason_counts[step_name] = {
            str(k): int(v) for k, v in payload.get("reason_counts", {}).items()
        }

    final_output_path = build_step_path(input_path, "step_6.fineweb_quality_filtered.feather")
    final_kept_rows = int(result["final_rows"])

    print("=" * 100)
    print(f"Step 7 dedup ({final_kept_rows} rows)")
    print("=" * 100)
    dedup_feather_path = build_step_path(input_path, "step_7.dedup_filtered.feather")
    dedup_removed_path = build_step_path(input_path, "step_7.dedup_removed.json")
    dedup_summary = run_dedup_step(
        result["final_df"],
        output_feather_path=dedup_feather_path,
        removed_json_path=dedup_removed_path,
        input_path_for_log=final_output_path,
        use_url_dedup=not args.no_url_dedup,
        use_title_dedup=not args.no_title_dedup,
        title_window_days=args.title_window_days,
        minhash_ngram_size=args.dedup_ngram_size,
        minhash_num_buckets=args.dedup_num_buckets,
        minhash_hashes_per_bucket=args.dedup_hashes_per_bucket,
        minhash_similarity_threshold=args.dedup_similarity,
        preview_chars=args.preview_chars,
    )
    final_output_path = dedup_feather_path
    final_kept_rows = int(dedup_summary["kept_rows"])
    step_names.append("step_7")
    per_step_reason_counts["step_7"] = {
        str(k): int(v)
        for k, v in load_removed_payload(dedup_removed_path).get("reason_counts", {}).items()
    }
    print(f"Step 7 kept/removed   : {dedup_summary['kept_rows']}/{dedup_summary['removed_rows']}")

    total_removed = int(result["source_rows"]) - final_kept_rows

    print("\n" + "=" * 100)
    print("FineWeb-style cleaning completed")
    print("=" * 100)
    print(f"Input file            : {input_path}")
    print(f"Final output file     : {final_output_path}")
    print(f"Total/Input rows      : {result['source_rows']}")
    print(f"Total/Kept rows       : {final_kept_rows}")
    print(f"Total/Removed rows    : {total_removed}")
    print("Reason counts by step:")
    for step_name in step_names:
        print(f"  {step_name}: {format_reason_counts(per_step_reason_counts.get(step_name, {}))}")


# Deduplication within each input file or crawl-date batch


URL_TRACKING_PARAM_PATTERN = re.compile(r"^(utm_|fbclid|gclid|mc_cid|mc_eid|igshid|spm)", re.IGNORECASE)
TITLE_PREFIX_STRIP_PATTERN = re.compile(
    r"^\s*(breaking|update|updated|exclusive|live|watch|video|photos?)\s*[:\-–|]\s*",
    re.IGNORECASE,
)


def canonicalize_url(url: object) -> str:
    """Normalize HTTP(S) URLs, removing fragments, tracking fields, and default ports."""
    raw = "" if url is None else str(url).strip()
    if not raw:
        return ""
    try:
        parsed = urlparse(raw)
    except ValueError:
        return raw.lower()
    if not parsed.scheme or not parsed.netloc:
        return raw.lower()

    raw_scheme = parsed.scheme.lower()
    if raw_scheme not in {"http", "https"}:
        return raw.lower()
    scheme = "https"
    netloc = parsed.netloc.lower()
    if netloc.endswith(":80") or netloc.endswith(":443"):
        netloc = netloc.rsplit(":", 1)[0]

    path = parsed.path or "/"
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/")
    if not path:
        path = "/"

    if parsed.query:
        kept = []
        for item in parsed.query.split("&"):
            if not item:
                continue
            key = item.split("=", 1)[0]
            if URL_TRACKING_PARAM_PATTERN.match(key):
                continue
            kept.append(item)
        query = "&".join(kept)
    else:
        query = ""

    return urlunparse((scheme, netloc, path, "", query, ""))


def normalize_title_for_hash(title: object) -> str:
    """Strip teaser prefixes, lowercase, and retain alphanumeric tokens."""
    raw = "" if title is None else str(title)
    raw = raw.strip()
    if not raw:
        return ""
    raw = TITLE_PREFIX_STRIP_PATTERN.sub("", raw)
    raw = raw.lower()
    tokens = re.findall(r"[a-z0-9]+", raw)
    return " ".join(tokens)


def title_hash(title: object) -> str:
    normalized = normalize_title_for_hash(title)
    if not normalized:
        return ""
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


# Match the structured-timestamp window in corpus.dates.
_PUBLISH_YEAR_MIN = 2010
_PUBLISH_YEAR_MAX = 2030


def _is_sane_publish_year(value: Any) -> bool:
    """Check whether a timestamp lies in the allowed publication-year window."""
    if value is None or value == "":
        return False
    import pandas as pd

    try:
        ts = pd.to_datetime(value, errors="coerce", utc=True)
    except (ValueError, OverflowError, OSError):
        return False
    if pd.isna(ts):
        return False
    return _PUBLISH_YEAR_MIN <= ts.year <= _PUBLISH_YEAR_MAX


def _resolve_publish_timestamp(row: dict[str, Any]):
    """Return the first valid publication timestamp, or pandas.NaT."""
    import pandas as pd

    for column in ("publish_date", "published", "date", "datetime", "timestamp"):
        if column in row and row[column] is not None and row[column] != "":
            try:
                ts = pd.to_datetime(row[column], errors="coerce", utc=True)
            except (ValueError, OverflowError, OSError):
                continue
            if pd.notna(ts) and _PUBLISH_YEAR_MIN <= ts.year <= _PUBLISH_YEAR_MAX:
                return ts
    return pd.NaT


def run_url_dedup(df, *, preview_chars: int) -> tuple[Any, list[dict[str, Any]]]:
    """Drop rows whose canonical URL has already been seen. Keep first occurrence."""
    if "url" not in df.columns or len(df) == 0:
        return df.reset_index(drop=True), []

    working = df.copy()
    working["__canonical_url"] = working["url"].map(canonicalize_url)

    seen: dict[str, int] = {}
    keep_mask: list[bool] = []
    removed_records: list[dict[str, Any]] = []

    for idx, row in tqdm(
        working.iterrows(),
        total=len(working),
        desc="step_7a url-dedup",
        unit="row",
    ):
        canonical = row["__canonical_url"]
        if not canonical:
            keep_mask.append(True)
            continue
        if canonical in seen:
            keep_mask.append(False)
            removed_records.append({
                "row_index": int(row.get("row_index", idx)),
                "url": None if row.get("url") is None else str(row.get("url")),
                "removal_step": "step_7",
                "removal_reason": "duplicate_canonical_url",
                "filter_name": "url_dedup",
                "canonical_url": canonical,
                "kept_row_index": int(seen[canonical]),
                "title_preview": truncate_text(row.get("title"), preview_chars),
                "text_preview": truncate_text(row.get("text"), preview_chars),
            })
        else:
            seen[canonical] = int(row.get("row_index", idx))
            keep_mask.append(True)

    kept = working[keep_mask].drop(columns=["__canonical_url"]).reset_index(drop=True)
    return kept, removed_records


def run_title_hash_dedup(
    df,
    *,
    window_days: int,
    preview_chars: int,
) -> tuple[Any, list[dict[str, Any]]]:
    """Keep the earliest matching title within each window.

    Missing dates fall back to retaining the first occurrence."""
    import pandas as pd

    if len(df) == 0:
        return df.reset_index(drop=True), []

    working = df.copy()
    working["__title_hash"] = working["title"].map(title_hash) if "title" in working.columns else ""
    timestamps = [_resolve_publish_timestamp(r) for r in working.to_dict("records")]
    working["__publish_ts"] = timestamps

    order = working.sort_values(
        by=["__publish_ts"], kind="mergesort", na_position="last"
    ).index.tolist()

    window = pd.Timedelta(days=window_days)
    last_kept: dict[str, Any] = {}
    keep_indices: set = set()
    removed_records: list[dict[str, Any]] = []

    for idx in tqdm(order, total=len(order), desc="step_7b title-dedup", unit="row"):
        row = working.loc[idx]
        h = row["__title_hash"]
        if not h:
            keep_indices.add(idx)
            continue
        ts = row["__publish_ts"]
        prev = last_kept.get(h)
        is_dup = False
        if prev is not None:
            prev_ts, prev_row_index = prev
            if pd.isna(ts) or pd.isna(prev_ts):
                is_dup = True
            elif (ts - prev_ts) <= window:
                is_dup = True
        if is_dup:
            removed_records.append({
                "row_index": int(row.get("row_index", idx)),
                "url": None if row.get("url") is None else str(row.get("url")),
                "removal_step": "step_7",
                "removal_reason": "duplicate_title_hash",
                "filter_name": "title_hash_dedup",
                "title_hash": h,
                "window_days": int(window_days),
                "kept_row_index": int(prev[1]),
                "title_preview": truncate_text(row.get("title"), preview_chars),
                "text_preview": truncate_text(row.get("text"), preview_chars),
            })
        else:
            keep_indices.add(idx)
            last_kept[h] = (ts, int(row.get("row_index", idx)))

    kept = (
        working.loc[sorted(keep_indices)]
        .drop(columns=["__title_hash", "__publish_ts"])
        .reset_index(drop=True)
    )
    return kept, removed_records


def _hash_5grams(text: str, n: int, num_hashes: int, seeds: list[int]) -> list[int]:
    """Return per-hash minimum hash value over n-grams of word-tokenized text."""
    words = re.findall(r"\w+", text.lower())
    if len(words) < n:
        return [0] * num_hashes
    ngrams = [" ".join(words[i:i + n]) for i in range(len(words) - n + 1)]
    minima = [0xFFFFFFFFFFFFFFFF] * num_hashes
    for ngram in ngrams:
        base = hashlib.blake2b(ngram.encode("utf-8"), digest_size=8).digest()
        base_int = int.from_bytes(base, "big")
        for h_idx, seed in enumerate(seeds):
            mixed = (base_int ^ seed) & 0xFFFFFFFFFFFFFFFF
            mixed = (mixed * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
            mixed ^= (mixed >> 27)
            mixed = (mixed * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
            if mixed < minima[h_idx]:
                minima[h_idx] = mixed
    return minima


# Initialize signature parameters once per worker.
_SIG_CTX: dict[str, Any] = {}


def _sig_init(ngram_size: int, num_hashes: int, seeds: list[int]) -> None:
    _SIG_CTX["ngram_size"] = ngram_size
    _SIG_CTX["num_hashes"] = num_hashes
    _SIG_CTX["seeds"] = seeds


def _sig_worker(text: Any) -> list[int]:
    """Compute one signature with the shared worker parameters."""
    return _hash_5grams(
        "" if text is None else str(text),
        _SIG_CTX["ngram_size"], _SIG_CTX["num_hashes"], _SIG_CTX["seeds"],
    )


def run_minhash_dedup(
    df,
    *,
    ngram_size: int,
    num_buckets: int,
    hashes_per_bucket: int,
    similarity_threshold: float,
    preview_chars: int,
    seed: int = 0xC0FFEE,
) -> tuple[Any, list[dict[str, Any]]]:
    """Cluster body n-grams with MinHash LSH.

    similarity_threshold is reported only; bands set the match probability."""
    import random as _random

    if len(df) == 0:
        return df.reset_index(drop=True), []

    num_hashes = num_buckets * hashes_per_bucket
    rng = _random.Random(seed)
    seeds = [rng.getrandbits(64) for _ in range(num_hashes)]

    # Ordered mapping and shared seeds keep signatures independent of worker count.
    texts = df["text"].tolist()
    sig_workers = int(os.environ.get("MINHASH_WORKERS", "1") or "1")
    if sig_workers > 1 and len(texts) > 1:
        from concurrent.futures import ProcessPoolExecutor

        chunksize = max(1, len(texts) // (sig_workers * 8))
        signatures = [None] * len(texts)
        with ProcessPoolExecutor(
            max_workers=sig_workers,
            initializer=_sig_init,
            initargs=(ngram_size, num_hashes, seeds),
        ) as pool:
            for i, sig in enumerate(tqdm(
                pool.map(_sig_worker, texts, chunksize=chunksize),
                total=len(texts), desc=f"step_7c minhash sig (x{sig_workers})", unit="doc",
            )):
                signatures[i] = sig
    else:
        signatures = []
        for text in tqdm(texts, desc="step_7c minhash sig", unit="doc"):
            signatures.append(_hash_5grams("" if text is None else str(text), ngram_size, num_hashes, seeds))

    bucket_maps: list[dict[tuple, list[int]]] = [dict() for _ in range(num_buckets)]
    for doc_idx, sig in enumerate(signatures):
        for bucket_idx in range(num_buckets):
            start = bucket_idx * hashes_per_bucket
            band = tuple(sig[start:start + hashes_per_bucket])
            bucket_maps[bucket_idx].setdefault(band, []).append(doc_idx)

    parent = list(range(len(signatures)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for bucket_map in bucket_maps:
        for members in bucket_map.values():
            if len(members) < 2:
                continue
            anchor = members[0]
            for other in members[1:]:
                union(anchor, other)

    clusters: dict[int, list[int]] = {}
    for doc_idx in range(len(signatures)):
        clusters.setdefault(find(doc_idx), []).append(doc_idx)

    keep_indices: set[int] = set()
    removed_records: list[dict[str, Any]] = []

    df_records = df.to_dict("records")

    # Keep the earliest date; break ties by row index and put missing dates last.
    def _earliest_member(members: list[int]) -> int:
        import pandas as pd

        def _key(idx: int):
            ts = _resolve_publish_timestamp(df_records[idx])
            row_index = int(df_records[idx].get("row_index", idx))
            # POSIX seconds avoid nanosecond timestamp overflow.
            ts_key = float("inf") if pd.isna(ts) else ts.timestamp()
            return (ts_key, row_index)

        return min(members, key=_key)

    for root, members in clusters.items():
        if len(members) == 1:
            keep_indices.add(members[0])
            continue
        chosen = _earliest_member(members)
        keep_indices.add(chosen)
        kept_row_index = int(df_records[chosen].get("row_index", chosen))
        kept_url = df_records[chosen].get("url")
        for other in members:
            if other == chosen:
                continue
            other_row = df_records[other]
            removed_records.append({
                "row_index": int(other_row.get("row_index", other)),
                "url": None if other_row.get("url") is None else str(other_row.get("url")),
                "removal_step": "step_7",
                "removal_reason": "minhash_near_duplicate",
                "filter_name": "minhash_dedup",
                "cluster_size": len(members),
                "cluster_id": int(root),
                "kept_row_index": kept_row_index,
                "kept_url": None if kept_url is None else str(kept_url),
                "ngram_size": int(ngram_size),
                "num_buckets": int(num_buckets),
                "hashes_per_bucket": int(hashes_per_bucket),
                "similarity_threshold": float(similarity_threshold),
                "title_preview": truncate_text(other_row.get("title"), preview_chars),
                "text_preview": truncate_text(other_row.get("text"), preview_chars),
            })

    kept_df = df.iloc[sorted(keep_indices)].reset_index(drop=True)
    return kept_df, removed_records


def run_dedup_step(
    df,
    *,
    output_feather_path: Path,
    removed_json_path: Path,
    input_path_for_log: Path,
    use_url_dedup: bool,
    use_title_dedup: bool,
    title_window_days: int,
    minhash_ngram_size: int,
    minhash_num_buckets: int,
    minhash_hashes_per_bucket: int,
    minhash_similarity_threshold: float,
    preview_chars: int,
) -> dict[str, Any]:
    """Apply 7a → 7b → 7c, write artifacts, return summary."""
    pre_count = int(len(df))
    current = df.reset_index(drop=True).copy() if len(df) else df
    all_removed: list[dict[str, Any]] = []

    if use_url_dedup and len(current):
        current, removed = run_url_dedup(current, preview_chars=preview_chars)
        all_removed.extend(removed)

    if use_title_dedup and len(current):
        current, removed = run_title_hash_dedup(
            current, window_days=title_window_days, preview_chars=preview_chars
        )
        all_removed.extend(removed)

    if len(current):
        current, removed = run_minhash_dedup(
            current,
            ngram_size=minhash_ngram_size,
            num_buckets=minhash_num_buckets,
            hashes_per_bucket=minhash_hashes_per_bucket,
            similarity_threshold=minhash_similarity_threshold,
            preview_chars=preview_chars,
        )
        all_removed.extend(removed)

    output_feather_path.parent.mkdir(parents=True, exist_ok=True)
    current.to_feather(output_feather_path)

    save_removed_json(
        removed_json_path,
        step_name="step_7",
        filter_name="dedup",
        input_file=input_path_for_log,
        output_file=output_feather_path,
        removed_records=all_removed,
        kept_count=int(len(current)),
    )

    return {
        "output_path": output_feather_path,
        "removed_output_path": removed_json_path,
        "input_rows": pre_count,
        "kept_rows": int(len(current)),
        "removed_rows": len(all_removed),
        "final_df": current,
    }


# Pipeline inspection


DEFAULT_INSPECT_INPUT = DEFAULT_PIPELINE_INPUT
INSPECT_N_KEPT = 100
INSPECT_N_REMOVED = 0
INSPECT_PREVIEW = 10000


def _inspect_stem(input_path: Path, suffix: str) -> Path:
    return input_path.with_name(f"{input_path.stem}.{suffix}")


def _inspect_show_kept(input_path: Path, n_kept: int, preview: int) -> None:
    import pandas as pd

    dedup_path = _inspect_stem(input_path, "step_7.dedup_filtered.feather")
    step6_path = _inspect_stem(input_path, "step_6.fineweb_quality_filtered.feather")
    path = dedup_path if dedup_path.exists() else step6_path
    df = pd.read_feather(path)
    print("=" * 100)
    print(f"KEPT (final): {len(df)} rows from {path.name}")
    print("=" * 100)
    sample = df.sample(min(n_kept, len(df)), random_state=0)
    for _, row in sample.iterrows():
        print(f"\n--- row_index={row.get('row_index')}  url={row.get('url')}")
        print(f"TITLE  : {row.get('title')}")
        print(f"SUMMARY: {str(row.get('summary'))[:200]}")
        print(f"TEXT   : {str(row.get('text'))[:preview]}")
        print("===============================================")


def _inspect_show_removed(input_path: Path, n_removed: int, preview: int) -> None:
    steps = [
        ("step_1", "step_1.removed.json"),
        ("step_2", "step_2.language_removed.json"),
        ("step_3", "step_3.gopher_repetition_removed.json"),
        ("step_4", "step_4.gopher_quality_removed.json"),
        ("step_5", "step_5.c4_removed.json"),
        ("step_6", "step_6.fineweb_quality_removed.json"),
        ("step_7", "step_7.dedup_removed.json"),
    ]
    for step_name, suffix in steps:
        path = _inspect_stem(input_path, suffix)
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        records = payload.get("removed_records", [])
        print("\n" + "=" * 100)
        print(f"REMOVED at {step_name}: {len(records)} rows  ({path.name})")
        print(f"reason_counts: {payload.get('reason_counts')}")
        print("=" * 100)
        by_reason: dict[str, list] = {}
        for r in records:
            by_reason.setdefault(str(r.get("removal_reason")), []).append(r)
        for reason, items in by_reason.items():
            for r in random.sample(items, min(n_removed, len(items))):
                print(
                    f"\n[reason={reason}] row_index={r.get('row_index')}  "
                    f"url={r.get('url')}"
                )
                print(f"  title  : {r.get('title_preview')}")
                print(f"  text   : {str(r.get('text_preview'))[:preview]}")
                for k in (
                    "predicted_language",
                    "language_confidence",
                    "trigger_lines",
                    "trigger_sentences",
                    "trigger_paragraphs",
                    "trigger_ngrams",
                    "kept_sentence_count_after_c4_line_cleaning",
                ):
                    if k in r:
                        print(f"  {k}: {r[k]}")


def cmd_inspect(args: argparse.Namespace) -> None:
    random.seed(0)
    input_path = args.input.expanduser().resolve() if args.input else DEFAULT_INSPECT_INPUT
    _inspect_show_kept(input_path, n_kept=args.n_kept, preview=args.preview)
    _inspect_show_removed(input_path, n_removed=args.n_removed, preview=args.preview)


# Cross-month date filtering and daily sharding


def _collect_reshard_inputs(input_root: Path) -> list[Path]:
    """Find monthly feathers, preferring flat files over nested pipeline outputs."""
    flat = sorted(input_root.glob("[0-9][0-9][0-9][0-9]-[0-9][0-9].feather"))
    if flat:
        return flat
    aggs: list[Path] = []
    for d in sorted(p for p in input_root.iterdir() if p.is_dir()):
        agg = d / f"{d.name}.aggregate.step_7.dedup_filtered.feather"
        if agg.exists():
            aggs.append(agg)
    return aggs


def _published_day_series(df, pd_mod):
    """Return resolved UTC days from the date column, or None for invalid dates."""
    col = "date" if "date" in df.columns else None
    if col is None:
        return pd_mod.Series([None] * len(df), index=df.index)
    ts = pd_mod.to_datetime(df[col], errors="coerce", utc=True)
    day = ts.dt.strftime("%Y-%m-%d")
    day = day.where(ts.notna(), None)
    return day


def cmd_reshard_daily(args: argparse.Namespace) -> None:
    import pandas as pd

    input_root = args.input_root.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    win_start, win_end = args.window_start, args.window_end
    if win_start > win_end:
        raise SystemExit(f"--window-start ({win_start}) must be <= --window-end ({win_end})")

    inputs = _collect_reshard_inputs(input_root)
    if not inputs:
        raise SystemExit(
            f"No per-month feathers under {input_root} "
            f"(looked for <month>.feather and <month>/*.aggregate.step_7...feather)."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 100)
    print("reshard-daily — global publish-date window + daily reshard")
    print(f"  input root   : {input_root}")
    print(f"  inputs       : {len(inputs)} month feathers")
    print(f"  window       : [{win_start}, {win_end}]  (inclusive, by published day)")
    print(f"  cross-dedup  : {'off' if args.no_cross_dedup else 'on (id then url, keep first)'}")
    print(f"  out dir      : {out_dir}")
    print("=" * 100)

    frames = []
    total_in = 0
    for p in tqdm(inputs, desc="reshard read", unit="file"):
        df = pd.read_feather(p)
        total_in += len(df)
        frames.append(df)
    union = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    del frames
    print(f"  concatenated : {total_in:,} rows from {len(inputs)} months")

    # Apply the inclusive window to resolved UTC publication days.
    day = _published_day_series(union, pd)
    union["__pub_day"] = day
    has_day = union["__pub_day"].notna()
    in_win = has_day & (union["__pub_day"] >= win_start) & (union["__pub_day"] <= win_end)
    dropped_nodate = int((~has_day).sum())
    dropped_oob = int((has_day & ~in_win).sum())
    union = union[in_win].reset_index(drop=True)
    print(f"  dropped (no/garbage date) : {dropped_nodate:,}")
    print(f"  dropped (outside window)  : {dropped_oob:,}")
    print(f"  kept in window            : {len(union):,}")

    # Cross-month dedup uses exact IDs and URLs, retaining the first capture.
    if not args.no_cross_dedup and len(union):
        before = len(union)
        if "id" in union.columns:
            union = union.drop_duplicates(subset=["id"], keep="first")
        if "url" in union.columns:
            nonempty = union["url"].fillna("").astype(str).str.strip() != ""
            dup_url = nonempty & union.duplicated(subset=["url"], keep="first")
            union = union[~dup_url]
        union = union.reset_index(drop=True)
        print(f"  cross-crawl duplicates dropped: {before - len(union):,}")

    # Drop row indices that were local to the original crawl month.
    days = union["__pub_day"]
    union = union.drop(columns=["__pub_day", "row_index"], errors="ignore")
    written = 0
    day_counts: dict[str, int] = {}
    for day_val, idx in tqdm(days.groupby(days).groups.items(),
                             desc="reshard write days", unit="day"):
        sub = union.loc[idx].reset_index(drop=True)
        out_path = out_dir / f"{day_val}.feather"
        tmp = out_dir / f".{day_val}.feather.part"
        sub.to_feather(tmp)
        os.replace(tmp, out_path)
        day_counts[day_val] = len(sub)
        written += 1

    print("=" * 100)
    print(f"reshard-daily done: wrote {written} day-shard feathers to {out_dir}")
    print(f"  total rows out : {sum(day_counts.values()):,}")
    if day_counts:
        lo, hi = min(day_counts), max(day_counts)
        print(f"  day range      : {lo} .. {hi}")
        print(f"  rows/day       : min={min(day_counts.values()):,} "
              f"max={max(day_counts.values()):,} "
              f"mean={sum(day_counts.values())//len(day_counts):,}")
    print(f"\nNext: embed the day shards, e.g.")
    print(f"  python3 -m corpus.embed --data-root {out_dir} \\")
    print(f"    --index-root faiss_index_daily --skip-existing")


# WARC extraction


def _iter_warc_responses(warc_path: Path) -> Iterable[dict[str, Any]]:
    """Yield response metadata and HTML, accepting records without Content-Type."""
    from warcio.archiveiterator import ArchiveIterator

    with open(warc_path, "rb") as fh:
        for record in ArchiveIterator(fh):
            if record.rec_type != "response":
                continue
            mime_type = (
                record.http_headers.get_header("Content-Type")
                if record.http_headers else None
            )

            # Missing Content-Type is accepted; explicit non-HTML types are skipped.

            if mime_type and "text/html" not in mime_type:
                continue
            try:
                payload = record.content_stream().read()
            except Exception:
                continue
            if not payload:
                continue
            yield {
                "id": record.rec_headers.get_header("WARC-Record-ID"),
                "url": record.rec_headers.get_header("WARC-Target-URI"),
                "date_crawled": record.rec_headers.get_header("WARC-Date"),
                "content_length": record.rec_headers.get_header("Content-Length"),
                "mime_type": mime_type,
                "html": payload,
            }


def _extract_article(html_bytes: bytes, date_crawled: str | None = None) -> dict[str, Any] | None:
    """Extract English article text and assign its structured-metadata UTC date."""
    import trafilatura

    from corpus.dates import extract_published_date

    try:
        data = trafilatura.bare_extraction(
            html_bytes,
            include_comments=False,
            deduplicate=True,
            with_metadata=True,
            target_language="en",
        )
    except Exception:
        return None
    if not data:
        return None


    if not isinstance(data, dict):
        data = data.as_dict() if hasattr(data, "as_dict") else dict(data.__dict__)

    text = (data.get("raw_text") or data.get("text") or "").strip()
    if not text:
        return None

    # Use structured metadata rather than trafilatura's date estimate.
    dres = extract_published_date(html_bytes, date_crawled)

    if not _is_sane_publish_year(dres.get("published_date")):
        return None


    return {
        "date": dres["published_date"],          
        "date_published": dres["date_published"],
        "date_modified": dres["date_modified"],
        "date_source": dres["date_source"],
        "date_trafilatura": data.get("date"),    
        "title": data.get("title"),
        "summary": data.get("description"),
        "hostname": data.get("hostname"),
        "text": text,
    }


def _parse_warc_worker(warc_path: Path) -> tuple[str, Path, Path | None, Any]:
    """Return parse status, paths, and statistics or an error string."""
    try:
        output_path, stats = _parse_one_warc(warc_path)
    except Exception as exc:
        import traceback
        return "error", warc_path, None, f"{exc!r}\n{traceback.format_exc()}"
    if output_path is None:
        return "empty", warc_path, None, stats
    return "ok", warc_path, output_path, stats


def _parse_one_warc(warc_path: Path) -> tuple[Path | None, dict[str, Any]]:
    """Atomically write extracted articles to a sibling feather file.

    Return (None, stats) when no usable rows remain."""
    import pandas as pd

    final_path = warc_path.with_name(warc_path.name.replace(".warc.gz", ".feather"))
    tmp_path = final_path.with_suffix(final_path.suffix + ".part")

    stats: dict[str, Any] = {
        "warc": str(warc_path),
        "responses_seen": 0,
        "rows_kept": 0,
        "extraction_failed": 0,
        "deduped_or_dropped": 0,
    }

    rows: list[dict[str, Any]] = []
    for record in _iter_warc_responses(warc_path):
        stats["responses_seen"] += 1
        try:
            article = _extract_article(record["html"], record["date_crawled"])
        except Exception:
            stats["extraction_failed"] += 1
            continue
        if article is None:
            stats["extraction_failed"] += 1
            continue
        rows.append({
            "id": record["id"],
            "url": record["url"],
            "date": article["date"],                      
            "date_crawled": record["date_crawled"],
            "date_published": article["date_published"],   
            "date_modified": article["date_modified"],     
            "date_source": article["date_source"],         
            "date_trafilatura": article["date_trafilatura"],  
            "title": article["title"],
            "summary": article["summary"],
            "hostname": article["hostname"],
            "text": article["text"],
            "content_length": record["content_length"],
            "mime_type": record["mime_type"],
        })

    if not rows:
        return None, stats

    df = (
        pd.DataFrame(rows)
        .dropna(subset=["text"])
        .drop_duplicates(subset=["text", "hostname"])
        .reset_index(drop=True)
    )
    stats["deduped_or_dropped"] = len(rows) - len(df)
    stats["rows_kept"] = len(df)

    if df.empty:
        return None, stats

    final_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_feather(tmp_path)
    os.replace(tmp_path, final_path)
    return final_path, stats


def cmd_parse(args: argparse.Namespace) -> None:
    from concurrent.futures import ProcessPoolExecutor, as_completed

    data_dir = args.data_dir.expanduser().resolve()
    if not data_dir.exists():
        raise SystemExit(f"--data-dir not found: {data_dir}")
    if not data_dir.is_dir():
        raise SystemExit(f"--data-dir is not a directory: {data_dir}")

    warcs = sorted(data_dir.glob("*.warc.gz"))
    if not warcs:
        raise SystemExit(f"No *.warc.gz files in {data_dir}")

    workers = max(1, int(args.workers))
    workers = min(workers, len(warcs))

    pending: list[Path] = []
    skipped = 0
    deleted = 0
    for warc in warcs:
        feather_path = warc.with_name(warc.name.replace(".warc.gz", ".feather"))
        if feather_path.exists() and not args.force:
            skipped += 1
            if not args.keep_warc and warc.exists():
                os.remove(warc)
                deleted += 1
            continue
        pending.append(warc)

    print("=" * 100)
    print(f"Parse WARCs in        : {data_dir}")
    print(f"WARC files            : {len(warcs)} (pending: {len(pending)}, skipped: {skipped})")
    print(f"Workers               : {workers}")
    print(f"Delete WARC on success: {not args.keep_warc}")
    print("=" * 100)

    parsed = 0
    empty = 0
    failed = 0
    total_rows = 0

    if not pending:
        print("\nNothing to do — all WARCs already have a sibling feather.")
    elif workers == 1:
        for warc in tqdm(pending, desc="parsing WARCs", unit="file"):
            try:
                output_path, stats = _parse_one_warc(warc)
            except Exception as exc:
                failed += 1
                tqdm.write(f"  ! {warc.name}: parse failed ({exc!r}) — leaving WARC in place")
                continue
            if output_path is None:
                empty += 1
                tqdm.write(
                    f"  - {warc.name}: 0 rows kept "
                    f"(responses={stats['responses_seen']}, "
                    f"extract_fail={stats['extraction_failed']}, "
                    f"deduped_or_dropped={stats['deduped_or_dropped']}) — leaving WARC in place"
                )
                continue
            parsed += 1
            total_rows += stats["rows_kept"]
            tqdm.write(
                f"  ✓ {warc.name} -> {output_path.name}  "
                f"rows={stats['rows_kept']} "
                f"(responses={stats['responses_seen']}, "
                f"extract_fail={stats['extraction_failed']}, "
                f"deduped_or_dropped={stats['deduped_or_dropped']})"
            )
            if not args.keep_warc:
                os.remove(warc)
                deleted += 1
    else:
        # Delete source archives only after workers publish their feather files.
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_parse_warc_worker, warc): warc for warc in pending}
            with tqdm(total=len(futures), desc="parsing WARCs", unit="file") as pbar:
                for future in as_completed(futures):
                    warc = futures[future]
                    try:
                        status, _wp, output_path, payload = future.result()
                    except Exception as exc:
                        failed += 1
                        tqdm.write(f"  ! {warc.name}: worker died ({exc!r})")
                        pbar.update(1)
                        continue

                    if status == "error":
                        failed += 1
                        tqdm.write(f"  ! {warc.name}: parse failed — leaving WARC in place\n{payload}")
                    elif status == "empty":
                        empty += 1
                        stats = payload
                        tqdm.write(
                            f"  - {warc.name}: 0 rows kept "
                            f"(responses={stats['responses_seen']}, "
                            f"extract_fail={stats['extraction_failed']}, "
                            f"deduped_or_dropped={stats['deduped_or_dropped']}) — leaving WARC in place"
                        )
                    else:
                        stats = payload
                        parsed += 1
                        total_rows += stats["rows_kept"]
                        tqdm.write(
                            f"  ✓ {warc.name} -> {output_path.name}  "
                            f"rows={stats['rows_kept']} "
                            f"(responses={stats['responses_seen']}, "
                            f"extract_fail={stats['extraction_failed']}, "
                            f"deduped_or_dropped={stats['deduped_or_dropped']})"
                        )
                        if not args.keep_warc and warc.exists():
                            os.remove(warc)
                            deleted += 1
                    pbar.update(1)

    print("\n" + "=" * 100)
    print("Parse summary")
    print("=" * 100)
    print(f"WARCs processed       : {len(warcs)}")
    print(f"Feathers written      : {parsed}")
    print(f"Already-present skips : {skipped}")
    print(f"Empty WARCs (no rows) : {empty}")
    print(f"Failed WARCs          : {failed}")
    print(f"WARCs deleted         : {deleted}")
    print(f"Total rows extracted  : {total_rows}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "CC-News text cleaning, language detection, and FineWeb-style filtering."
        )
    )
    sub = parser.add_subparsers(dest="command", required=True)


    p_clean = sub.add_parser(
        "clean",
        help=(
            "Normalize article text in a feather file: strip inline URLs, "
            "collapse whitespace, cap blank lines, and drop rows whose title "
            "or body becomes empty. Emits a cleaned feather plus diagnostic JSONs."
        ),
    )
    p_clean.add_argument("input", type=Path, help="Input feather file path.")
    p_clean.add_argument(
        "--output", type=Path, default=None,
        help="Output feather file path. Defaults to <input_stem>.step_1.feather.",
    )
    p_clean.add_argument(
        "--compare-output", type=Path, default=None,
        help="Comparison JSON file path. Defaults to <input_stem>.step_1.compare.json.",
    )
    p_clean.add_argument(
        "--removed-output", type=Path, default=None,
        help="Removed-rows JSON file path. Defaults to <input_stem>.step_1.removed.json.",
    )
    p_clean.add_argument(
        "--max-blank-lines", type=int, default=2,
        help="Maximum number of consecutive blank lines to keep.",
    )
    p_clean.add_argument(
        "--show-examples", type=int, default=5,
        help="How many changed examples to print.",
    )
    p_clean.set_defaults(func=cmd_clean)


    p_lang = sub.add_parser(
        "language",
        help=(
            "Identify English articles with FastText lid.176. `detect` scores "
            "every row, attaches language + confidence, and writes a feather "
            "of high-confidence English rows. `export-en-json` dumps already-"
            "scored English rows to JSON, sorted by confidence, for review."
        ),
    )
    p_lang.add_argument(
        "--mode", choices=["detect", "export-en-json"], default="detect",
        help="`detect`: run FastText language detection. "
             "`export-en-json`: export English rows from a labeled feather file.",
    )
    p_lang.add_argument(
        "--input", type=Path, default=DEFAULT_LANG_INPUT,
        help=f"Input feather file for detect mode. Default: {DEFAULT_LANG_INPUT}",
    )
    p_lang.add_argument(
        "--filtered-input", type=Path, default=DEFAULT_LANG_FILTERED_INPUT,
        help=f"Input feather file for export-en-json mode. Default: {DEFAULT_LANG_FILTERED_INPUT}",
    )
    p_lang.add_argument(
        "--output", type=Path, default=None,
        help="Output path. detect: <input_stem>.language_filtered.feather. "
             "export-en-json: <filtered_input_stem>.en_sorted.json.",
    )
    p_lang.add_argument(
        "--model", type=Path, default=DEFAULT_LANG_MODEL,
        help=f"FastText language identification model. Default: {DEFAULT_LANG_MODEL}",
    )
    p_lang.add_argument("--batch-size", type=int, default=512, help="Prediction batch size.")
    p_lang.add_argument(
        "--preview-chars", type=int, default=500,
        help="Maximum number of characters kept in each JSON text preview.",
    )
    p_lang.set_defaults(func=cmd_language)


    p_pipe = sub.add_parser(
        "pipeline",
        help=(
            "Run the full document quality cascade end-to-end (text "
            "normalization -> English filtering -> Gopher repetition + "
            "quality -> C4 -> FineWeb quality). Each stage writes its kept "
            "feather and a removed-rows JSON with per-row reason codes. "
            "Use --date to run the cascade across every feather in a date "
            "folder and emit aggregate batch outputs."
        ),
    )
    p_pipe.add_argument(
        "input", type=Path, nargs="?", default=DEFAULT_PIPELINE_INPUT,
        help=f"Input feather file path. Default: {DEFAULT_PIPELINE_INPUT}",
    )
    p_pipe.add_argument(
        "--language-threshold", type=float, default=LANGUAGE_THRESHOLD,
        help=f"Language confidence threshold. Default: {LANGUAGE_THRESHOLD}",
    )
    p_pipe.add_argument(
        "--preview-chars", type=int, default=500,
        help="Preview length stored in removed JSON files. Default: 500",
    )
    p_pipe.add_argument(
        "--fast-order", action="store_true",
        help="Run GopherQualityFilter before GopherRepetitionFilter. "
             "Faster on corpora with many short/low-quality texts, but differs "
             "from FineWeb's original order.",
    )
    p_pipe.add_argument(
        "--date", type=str, default=None,
        help="Batch mode. Accepts YYYY-MM / YYYYMM (month), YYYYMMDD (day prefix "
             "filter inside month folder), or an absolute/relative folder path. "
             f"When set, runs on all source feather files in that folder under {DATA_ROOT}.",
    )
    p_pipe.add_argument(
        "--workers", type=int, default=os.cpu_count() or 1,
        help="Number of worker processes for --date batch mode. Each source "
             "feather is processed in its own child process. Set to 1 to run "
             "serially in the parent. Default: os.cpu_count().",
    )


    dedup_group = p_pipe.add_argument_group(
        "deduplication (step_7)",
        "URL canonicalization + title-hash dedup + MinHash LSH applied once "
        "after step_6 as the final stage of the cascade. In --date mode runs "
        "on the per-month aggregate; in single-file mode runs on that file's "
        "step_6 output. Individual sub-stages can be skipped with "
        "--no-url-dedup / --no-title-dedup; MinHash always runs.",
    )
    dedup_group.add_argument(
        "--no-url-dedup", action="store_true", default=False,
        help="Skip URL canonicalization pre-pass.",
    )
    dedup_group.add_argument(
        "--no-title-dedup", action="store_true", default=False,
        help="Skip title-hash dedup pre-pass.",
    )
    dedup_group.add_argument(
        "--title-window-days", type=int, default=TITLE_DEDUP_WINDOW_DAYS,
        help=f"Sliding window in days for title-hash dedup. Default: {TITLE_DEDUP_WINDOW_DAYS}",
    )
    dedup_group.add_argument(
        "--dedup-ngram-size", type=int, default=MINHASH_NGRAM_SIZE,
        help=f"Word n-gram size for MinHash shingles. Default: {MINHASH_NGRAM_SIZE}",
    )
    dedup_group.add_argument(
        "--dedup-num-buckets", type=int, default=MINHASH_NUM_BUCKETS,
        help=f"Number of LSH buckets. Default: {MINHASH_NUM_BUCKETS}",
    )
    dedup_group.add_argument(
        "--dedup-hashes-per-bucket", type=int, default=MINHASH_HASHES_PER_BUCKET,
        help=f"Hashes per LSH bucket. Default: {MINHASH_HASHES_PER_BUCKET}",
    )
    dedup_group.add_argument(
        "--dedup-similarity", type=float, default=MINHASH_SIMILARITY_THRESHOLD,
        help=f"Reported Jaccard similarity target (logged in removed JSON; "
             f"actual cutoff is set by --dedup-num-buckets and "
             f"--dedup-hashes-per-bucket). Default: {MINHASH_SIMILARITY_THRESHOLD}",
    )

    p_pipe.set_defaults(func=cmd_pipeline)


    p_inspect = sub.add_parser(
        "inspect",
        help=(
            "Eyeball a pipeline run: print a random sample of the final kept "
            "rows, then, for each stage, removed rows grouped by removal "
            "reason. Used to sanity-check filter behavior on a dataset."
        ),
    )
    p_inspect.add_argument(
        "--input", type=Path, default=None,
        help=f"Source feather (used to locate companion outputs). "
             f"Default: {DEFAULT_INSPECT_INPUT}",
    )
    p_inspect.add_argument(
        "--n-kept", type=int, default=INSPECT_N_KEPT,
        help=f"Number of kept rows to sample. Default: {INSPECT_N_KEPT}",
    )
    p_inspect.add_argument(
        "--n-removed", type=int, default=INSPECT_N_REMOVED,
        help=f"Number of removed rows to sample per reason. Default: {INSPECT_N_REMOVED}",
    )
    p_inspect.add_argument(
        "--preview", type=int, default=INSPECT_PREVIEW,
        help=f"Max chars of text preview per row. Default: {INSPECT_PREVIEW}",
    )
    p_inspect.set_defaults(func=cmd_inspect)


    p_parse = sub.add_parser(
        "parse",
        help=(
            "Extract dated English articles into a feather file beside each WARC. "
            "Delete the WARC after success unless --keep-warc is set."
        ),
    )
    p_parse.add_argument(
        "--data-dir", type=Path, required=True,
        help="Directory containing *.warc.gz files (e.g. data/2025-06).",
    )
    p_parse.add_argument(
        "--keep-warc", action="store_true", default=False,
        help="Do not delete the source .warc.gz after a successful parse.",
    )
    p_parse.add_argument(
        "--force", action="store_true", default=False,
        help="Re-parse WARCs even if a sibling .feather already exists.",
    )
    p_parse.add_argument(
        "--workers", type=int, default=os.cpu_count() or 1,
        help="Parallel WARC workers; 1 runs in the parent process. Default: os.cpu_count().",
    )
    p_parse.set_defaults(func=cmd_parse)

    p_reshard = sub.add_parser(
        "reshard-daily",
        help=(
            "GLOBAL publish-date window + daily reshard (cross-month). Reads the "
            "per-month step_7 outputs, keeps rows whose published_date is in "
            "[--window-start, --window-end], optionally drops cross-crawl "
            "duplicates, and writes one <YYYY-MM-DD>.feather per publish day for "
            "per-day FAISS index building. Run ONCE after all months' pipeline "
            "runs finish."
        ),
    )
    p_reshard.add_argument(
        "--input-root", type=Path, required=True,
        help="Dir of collected per-month feathers (<month>.feather) or month "
             "dirs holding <month>.aggregate.step_7.dedup_filtered.feather.",
    )
    p_reshard.add_argument(
        "--out-dir", type=Path, required=True,
        help="Output dir for the <YYYY-MM-DD>.feather day shards.",
    )
    p_reshard.add_argument(
        "--window-start", type=str, required=True,
        help="Inclusive lower bound on published_date, YYYY-MM-DD (e.g. 2025-03-01).",
    )
    p_reshard.add_argument(
        "--window-end", type=str, required=True,
        help="Inclusive upper bound on published_date, YYYY-MM-DD (e.g. 2026-05-31).",
    )
    p_reshard.add_argument(
        "--no-cross-dedup", action="store_true", default=False,
        help="Skip the id/url cross-crawl dedup (keep same-article reposts that "
             "appear in multiple crawl-months).",
    )
    p_reshard.set_defaults(func=cmd_reshard_daily)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
