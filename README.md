# Forecast-Dojo

**Replayable Environments for Benchmarking and Training LLM Forecasting Agents**

## Setup

Python 3.10+. Run from the repository root. Install vLLM separately for embedding
and local model serving; the Python tool requires Linux namespace support.

```bash
python -m pip install -e ./strands-env -e '.[corpus,evidence-check]'

export ANTHROPIC_API_KEY="your-anthropic-api-key"
export OPENAI_API_KEY="your-openai-api-key"
export FORECAST_INDEX_ROOT=/path/to/news-index
```

## 1. Corpus construction

[src/corpus](src/corpus) prepares CC-News from March 2025 through May 2026:

```text
download → filter → regroup by publication month → embed
```

Regroup articles by their resolved UTC publication month, not crawl month.
Qwen3-Embedding-8B produces monthly FAISS shards containing `faiss.index`,
`texts.parquet`, and `metadata.json`. News, indexes, and checkpoints are not included.

```bash
bash scripts/build_corpus.sh download --help
bash scripts/build_corpus.sh filter --help
bash scripts/build_corpus.sh embed --help
```

Start the query embedding service in a separate terminal before evidence retrieval:

```bash
EMBEDDING_MODEL_PATH=/path/to/Qwen3-Embedding-8B \
bash scripts/serve_embeddings.sh
```

## 2. Event construction

[src/data](src/data) collects Polymarket events and applies market-side filters:

```text
collect_events → collect_prices → collect_volume → split
→ filter_lifetime → filter_activity → filter_signal
```

Run each module with `python -m data.<module> --help` for its input/output options.
`split` retains events whose full lifetimes fall within the requested time window.

## 3. Evidence check

[src/data/evidence_check](src/data/evidence_check) retrieves evidence and checks
sufficiency and leakage:

```text
build_bm25 → decompose → retrieve → judge → select_evidence
```

Run each module with `python -m data.evidence_check.<module> --help`.
Retrieval combines dense and BM25 results; `select_evidence` keeps sufficient events.

## 4. Forecast schedules

Create dated tasks from the selected events:

```bash
python -m data.build_dataset \
  --input /path/to/selected-events.jsonl --output outputs/tasks.jsonl \
  --auto-k --news-volume --index-root "$FORECAST_INDEX_ROOT"
```

Training and evaluation data are hosted on
[Hugging Face](https://huggingface.co/datasets/foye107/Forecast-Dojo).
Each row contains one event, its forecast dates, resolved label, and market references.

| Split | Events | Event–date tasks |
| --- | ---: | ---: |
| Train | 1,338 | 5,325 |
| Eval | 230 | 797 |

Download both splits from the repository root before running the benchmark:

```bash
hf download foye107/Forecast-Dojo data/train.jsonl data/eval.jsonl \
  --repo-type dataset --local-dir .
```

## 5. Benchmark

Model settings are in [configs/eval](configs/eval). Research modes require the
news index and embedding service; `no-tools` requires neither.

| Mode | Tools | Memory between dates |
| --- | --- | --- |
| `memory-free` | Search, read, Python | None |
| `memory-on` | Search, read, Python | Previous belief notebook |
| `no-tools` | None | None |

```bash
bash scripts/evaluate.sh memory-free --config configs/eval/gpt-5.4.json
bash scripts/evaluate.sh memory-on --config configs/eval/gpt-5.4.json
bash scripts/evaluate.sh no-tools --config configs/eval/gpt-5.4.json
```

Defaults: four rollouts per event–date, 120 tool rounds and 400 calls per date.
Output caps are 32,768 tokens with tools and 65,536 without. Use
`--questions-path data/train.jsonl` to collect training trajectories.

For local models, use [scripts/serve_model.sh](scripts/serve_model.sh) with
`MODEL_PATH`, a `MODEL_NAME` matching the profile's `model_id`, and the model's
`TOOL_CALL_PARSER`.

### Scoring

Results are saved to `results/<model>.<mode>.jsonl`, alongside run configuration
and token usage. Score a completed run with:

```bash
python -m evaluation.metrics results/gpt-5.4.memory-free.jsonl \
  --dataset data/eval.jsonl --output results/gpt-5.4.memory-free.scores.json
```

Metrics include Brier, accuracy, log loss, Information-alpha, calibration error,
and event-bootstrap confidence intervals.

## Structure

```text
src/corpus/                # News corpus and indexes
src/data/                  # Events and forecast schedules
src/data/evidence_check/    # Evidence screening
src/environment/           # Episodes, tools, and rewards
src/evaluation/            # Benchmark runner and metrics
scripts/                   # Launchers
configs/eval/              # Model settings
prompts/                   # Agent and screening prompts
strands-env/               # Shared agent runtime
```

Strands-derived code retains its [Apache-2.0 license](strands-env/LICENSE);
the subprocess interpreter retains CAMEL-AI and Slime attribution.
