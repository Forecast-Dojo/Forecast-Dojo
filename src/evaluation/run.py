"""Run forecasting agents on a fixed historical event schedule."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from prompts import load_prompt

from . import cost as cost_hook
from .metrics import format_scorecard, outcome_labels, score_forecast, score_run

logger = logging.getLogger(__name__)

DEFAULT_SAMPLING_PARAMS = {'max_new_tokens': 32768}
_REWARD_KEYS = (
    'reward', 'metric', 'proper_score', 'format_penalty', 'brier', 'log_prob',
    'accuracy', 'p_truth', 'parse_ok', 'prob_sum_raw', 'truth_in_forecast',
    'info_alpha', 'crowd_prob', 'format_quality', 'format_answer_ok', 'format_notebook_ok',
)


@dataclass
class Episode:
    """A question and its scheduled forecasts and scoring references."""

    event_id: str
    slug: str | None
    belief_kind: str
    label: str
    start_date: str
    question: dict[str, Any]
    forecast_dates: list[str]
    crowd_probs: list[float | None]
    domain: str | None


async def prewarm(env: ForecastEnv, episodes: list[Episode]) -> None:
    """Load shards spanning the scheduled retrieval windows."""
    if getattr(env, 'no_tools', False) or env.retriever_toolkit is None:
        logger.info('no-tools run: no retriever to prewarm')
        return
    earliest = min((ep.start_date for ep in episodes), default=None)
    latest = max((d for ep in episodes for d in ep.forecast_dates), default=None)
    if not (earliest and latest):
        return
    env.retriever_toolkit.start_date = earliest
    env.retriever_toolkit.cutoff_date = latest
    logger.info('prewarming FAISS shards in [%s, %s] ...', earliest, latest)
    n = await env.retriever_toolkit.prewarm()
    logger.info('prewarmed %d shard(s)', n)


def extract_trace(messages: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Serialize conversation text, tool calls, and tool results."""
    out: list[dict[str, Any]] = []
    turn = 0
    for msg in messages or []:
        role = msg.get('role')
        content = msg.get('content') or []
        if role == 'assistant':
            turn += 1
            texts = [b['text'] for b in content if isinstance(b, dict) and 'text' in b]
            tool_uses = [
                {
                    'id': tu.get('toolUseId') or '',
                    'name': tu.get('name'),
                    'input': tu.get('input') or {},
                }
                for b in content
                if isinstance(b, dict) and (tu := b.get('toolUse'))
            ]
            out.append({
                'role': 'assistant', 'turn': turn,
                'text': '\n'.join(texts), 'tool_uses': tool_uses,
            })
        elif role == 'user':
            texts = [b['text'] for b in content if isinstance(b, dict) and 'text' in b]
            tool_results = []
            for b in content:
                if isinstance(b, dict) and (tr := b.get('toolResult')):
                    pieces = [
                        c['text'] for c in tr.get('content') or []
                        if isinstance(c, dict) and 'text' in c
                    ]
                    tool_results.append({
                        'id': tr.get('toolUseId') or '',
                        'status': tr.get('status', 'success'),
                        'text': '\n'.join(pieces),
                    })
            out.append({
                'role': 'user', 'turn': turn if tool_results else 0,
                'text': '\n'.join(texts), 'tool_results': tool_results,
            })
    return out


def resume_key(episode: Episode, rollout_idx: int) -> tuple[str, str | None, int]:
    """Identify an episode by event, first date, and rollout."""
    first_date = episode.forecast_dates[0] if episode.forecast_dates else None
    return str(episode.event_id), first_date, rollout_idx


def load_completed(path: Path) -> set[tuple[str, str | None, int]]:
    """Read resume keys from recorded initial steps."""
    done: set[tuple[str, str | None, int]] = set()
    if not path.exists():
        return done
    with path.open(encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                if int(row.get('step_idx', 0)) == 0:
                    done.add((
                        str(row.get('event_id')),
                        row.get('forecast_date'),
                        int(row.get('rollout_idx', 0)),
                    ))
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
    return done


def python_sandbox_state() -> str:
    """Read the Python tool's sandbox setting."""
    return (
        'off'
        if os.environ.get('FORECAST_PYTHON_SANDBOX', 'on').strip().lower()
        in ('0', 'off', 'false', 'no')
        else 'on'
    )


def load_episodes(
    path: Path,
    offset: int = 0,
    limit: int | None = None,
    mode: str = 'memory-free',
) -> tuple[list[Episode], ForecastMode]:
    """Load fixed schedules, splitting memory-free events into one episode per date."""
    from environment.env import ForecastMode

    forecast_mode = ForecastMode.MEMORY_ON if mode == 'memory-on' else ForecastMode.MEMORY_FREE
    rows = [
        json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f'{path}: no event rows')
    episodes = []
    end = None if limit is None else offset + limit
    for row in rows[offset:end]:
        meta = row['metadata']
        dates = list(meta['forecast_dates']) if 'forecast_dates' in meta else [meta['forecast_date']]
        crowd = (
            list(meta.get('crowd_probs') or [])
            if 'forecast_dates' in meta else [meta.get('crowd_prob')]
        )
        if not dates or dates != sorted(dates) or len(set(dates)) != len(dates):
            raise ValueError(f"{path}: {row.get('event_id')} has empty, unordered, or duplicate forecast dates")
        if crowd and len(crowd) != len(dates):
            raise ValueError(f'{path}: crowd probabilities do not match forecast dates')
        crowd = crowd or [None] * len(dates)
        common = dict(
            event_id=str(row['event_id']),
            slug=row.get('slug'),
            belief_kind=row['belief_kind'],
            label=row['label'],
            start_date=meta['start_date'],
            question=meta['question'],
            domain=meta.get('domain'),
        )
        if forecast_mode.uses_notebook:
            episodes.append(Episode(**common, forecast_dates=dates, crowd_probs=crowd))
        else:
            episodes.extend(
                Episode(**common, forecast_dates=[date], crowd_probs=[prob])
                for date, prob in zip(dates, crowd)
            )
    return episodes, forecast_mode


def resolve_system_prompt(args: argparse.Namespace, mode: ForecastMode) -> str:
    if args.system_prompt_path:
        return Path(args.system_prompt_path).read_text(encoding='utf-8').strip()
    return load_prompt(mode.value, search=not args.no_tools, enable_code=not args.no_tools)


def build_env(args: argparse.Namespace, mode: ForecastMode, model_factory: Any) -> ForecastEnv:
    from environment.env import ForecastEnv

    return ForecastEnv(
        model_factory=model_factory, reward_fn=None, mode=mode.value,
        model_backend=args.backend,
        system_prompt=resolve_system_prompt(args, mode), index_root=args.index_root,
        embedding_endpoint=args.embedding_endpoint, embedding_model=args.embedding_model,
        enable_code=not args.no_tools, no_tools=args.no_tools,
        code_concurrency=2 * args.concurrency,
        retriever_concurrency=args.retriever_concurrency or max(32, 2 * args.concurrency),
        max_tool_iters=args.max_tool_iters, max_tool_calls=args.max_tool_calls,
        start_date='1970-01-01', cutoff_date='1970-01-01', verbose=False,
    )


async def run_rollout(
    env: ForecastEnv,
    reward: ForecastReward,
    episode: Episode,
    rollout_idx: int,
    run_id: str,
    model_name: str,
    mode: ForecastMode,
    keep_trace: bool,
) -> list[dict[str, Any]]:
    """Run an episode and record each date's forecast and scores."""
    from environment.notebook import extract_answer

    harness = 'dojo'
    try:
        result = await env.run_episode(
            question_prompt=env.render_question(episode.question),
            forecast_dates=episode.forecast_dates,
            start_date=episode.start_date,
        )
    except Exception as e:
        logger.exception('episode failed: event_id=%s rollout=%d', episode.event_id, rollout_idx)
        return [{
            'run_id': run_id,
            'event_id': episode.event_id,
            'rollout_idx': rollout_idx,
            'harness': harness,
            'forecast_date': episode.forecast_dates[0] if episode.forecast_dates else None,
            'step_idx': 0,
            'error': f'{type(e).__name__}: {e}',
        }]
    rows: list[dict[str, Any]] = []
    for step_idx, step in enumerate(result.steps):
        crowd_prob = episode.crowd_probs[step_idx] if step_idx < len(episode.crowd_probs) else None
        rr = await reward.score_step(
            notebook=step.final_response, label=episode.label, crowd_prob=crowd_prob,
        )
        info = rr.info
        row: dict[str, Any] = {
            'run_id': run_id,
            'model_name': model_name,
            'mode': 'no-tools' if env.no_tools else mode.value,
            'harness': harness,
            'event_id': episode.event_id,
            'slug': episode.slug,
            'title': episode.question.get('title', ''),
            'belief_kind': episode.belief_kind,
            'domain': episode.domain,
            'rollout_idx': rollout_idx,
            'step_idx': step_idx,
            'forecast_date': step.forecast_date,
            'resolved_label': episode.label,
            'agent_forecast': extract_answer(step.final_response),
            'final_response': step.final_response,
            **{k: info.get(k) for k in _REWARD_KEYS},
            'termination': step.termination_reason.value,
            'metrics': step.metrics,
            'cost': cost_hook.trajectory_cost(step.metrics),
            'outcome_labels': outcome_labels(episode.question),
        }
        scores = score_forecast(
            row['agent_forecast'], episode.label, row['outcome_labels'],
            parse_ok=info.get('parse_ok') is not False, crowd_prob=crowd_prob,
        )
        row['evaluation_scores'] = {
            key: scores[key]
            for key in ('usable', 'brier', 'accuracy', 'log_loss', 'info_alpha', 'p_truth')
        }
        if keep_trace:
            row['agent_trace'] = extract_trace(step.messages)
        rows.append(row)
    return rows


_RESUME_CRITICAL = (
    'mode', 'backend', 'base_url', 'reasoning_effort', 'no_thinking', 'n_rollouts',
    'max_tool_iters', 'max_tool_calls', 'index_root', 'embedding_model', 'questions_path',
)


def assert_resume_compatible(
    output: Path,
    args: argparse.Namespace,
    model_id: str,
    sampling_params: dict[str, Any],
    system_prompt: str,
) -> None:
    if not output.exists() or not output.stat().st_size:
        return
    sidecar = output.with_suffix(output.suffix + '.manifest.json')
    if not sidecar.is_file():
        raise ValueError('An existing output requires its run manifest; use a new output file')
    old = json.loads(sidecar.read_text(encoding='utf-8'))
    changed = [
        key for key in _RESUME_CRITICAL
        if old.get('args', {}).get(key) != getattr(args, key, None)
    ]
    for key, value in (
        ('model_id', model_id),
        ('sampling_params', sampling_params),
        ('system_prompt', system_prompt),
        ('python_sandbox', python_sandbox_state()),
    ):
        if old.get(key) != value:
            changed.append(key)
    if changed:
        raise ValueError(f"Run configuration changed ({', '.join(changed)}); use a new output file")


def write_manifest(
    output: Path,
    args: argparse.Namespace,
    run_id: str,
    model_id: str,
    mode: ForecastMode,
    sampling_params: dict[str, Any],
    system_prompt: str,
) -> None:
    manifest = dict(
        run_id=run_id,
        mode=args.mode,
        model_id=model_id,
        python_sandbox=python_sandbox_state(),
        sampling_params=sampling_params,
        args=vars(args),
        system_prompt=system_prompt,
    )
    output.with_suffix(output.suffix + '.manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding='utf-8',
    )


async def amain(args: argparse.Namespace) -> None:
    """Evaluate scheduled episodes concurrently and save their results."""
    from tqdm.auto import tqdm
    from tqdm.contrib.logging import logging_redirect_tqdm
    from environment.reward import ForecastReward
    from .models import build_model_factory

    asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(
        max_workers=max(64, 4 * args.concurrency), thread_name_prefix='blocking',
    ))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    episodes, mode = load_episodes(Path(args.questions_path), args.offset, args.limit, args.mode)
    logger.info('loaded %d episode(s) [%s mode] from %s', len(episodes), mode.value, args.questions_path)
    sampling_params = dict(DEFAULT_SAMPLING_PARAMS)
    if args.sampling_params:
        for k, v in json.loads(args.sampling_params).items():
            if v is None:
                sampling_params.pop(k, None)
            else:
                sampling_params[k] = v
    if args.temperature is not None:
        sampling_params['temperature'] = args.temperature
    if args.max_new_tokens is not None:
        sampling_params['max_new_tokens'] = args.max_new_tokens
    model_factory, model_id = build_model_factory(args, sampling_params)
    cost_hook.set_default_model(
        model_id, args.backend,
        cache_write_ttl=(sampling_params.get('cache_control') or {}).get('ttl', '5m'),
    )
    model_factory = cost_hook.tap_factory(model_factory, output.with_suffix('.calls.jsonl'), model_id)
    model_name = args.model_name or Path(model_id).name
    run_id = args.run_id or f'{model_name}.{Path(args.questions_path).stem}.{args.mode}'
    env = build_env(args, mode, model_factory)
    assert_resume_compatible(output, args, model_id, sampling_params, env.system_prompt)
    write_manifest(output, args, run_id, model_id, mode, sampling_params, env.system_prompt)
    reward = ForecastReward(
        metric=args.reward_metric, format_coef=args.format_coef,
        score_notebook=mode.uses_notebook,
    )
    await prewarm(env, episodes)
    done = load_completed(output)
    units = [
        (ep, r) for ep in episodes for r in range(args.n_rollouts)
        if resume_key(ep, r) not in done
    ]
    logger.info(
        'dispatching %d rollout(s) (%d already done) at concurrency %d',
        len(units), len(done), args.concurrency,
    )
    sem = asyncio.Semaphore(max(1, args.concurrency))
    write_lock = asyncio.Lock()
    pbar = tqdm(total=len(units), desc='episodes', unit='ep')
    with output.open('a', encoding='utf-8') as out:

        async def worker(episode: Episode, rollout_idx: int) -> None:
            async with sem:
                rows = await run_rollout(
                    env, reward, episode, rollout_idx, run_id, model_name, mode,
                    keep_trace=not args.no_trace,
                )
            async with write_lock:
                out.write(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
                out.flush()
            pbar.update(1)

        with logging_redirect_tqdm():
            await asyncio.gather(*(worker(ep, r) for ep, r in units))
    pbar.close()
    await env.cleanup()
    logger.info('done -> %s', output)
    print(format_scorecard(score_run(
        output, dataset=Path(args.questions_path), n_rollouts=args.n_rollouts, bootstrap=0,
    )))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, help='JSON file supplying CLI defaults.')
    parser.add_argument('--questions-path', default='data/eval.jsonl')
    parser.add_argument('--output', help='Result JSONL; defaults to results/<model>.<mode>.jsonl.')
    parser.add_argument('--mode', choices=('memory-free', 'memory-on', 'no-tools'), default='memory-free')
    parser.add_argument('--offset', type=int, default=0, help='Skip this many event rows.')
    parser.add_argument('--limit', type=int, help='Maximum event rows.')
    parser.add_argument('--index-root', default=os.environ.get('FORECAST_INDEX_ROOT'), help='External monthly FAISS index directory.')
    parser.add_argument('--embedding-endpoint', default='http://localhost:8001/v1')
    parser.add_argument('--embedding-model', default='Qwen3-Embedding-8B')
    parser.add_argument('--retriever-concurrency', type=int)
    parser.add_argument(
        '--backend', choices=('anthropic', 'openai', 'openai-compatible', 'sglang'),
        default='openai',
    )
    parser.add_argument('--model-id')
    parser.add_argument('--model-name')
    parser.add_argument('--base-url', help='API endpoint; local OpenAI-compatible servers include /v1.')
    parser.add_argument('--tokenizer-path')
    parser.add_argument('--tool-parser', choices=('glm', 'hermes', 'qwen_xml', 'kimi_k2'))
    parser.add_argument('--max-connections', type=int, default=1000)
    parser.add_argument('--request-timeout', type=float, default=1200.0)
    parser.add_argument('--api-key-env', help='Override the selected backend\'s API key environment variable.')
    parser.add_argument('--no-thinking', action='store_true', help='Disable Claude thinking.')
    parser.add_argument('--reasoning-effort', choices=('none', 'low', 'medium', 'high', 'max', 'xhigh'))
    parser.add_argument('--sampling-params', help='JSON sampling parameters; null removes a default key.')
    parser.add_argument('--temperature', type=float)
    parser.add_argument('--max-new-tokens', type=int)
    parser.add_argument('--max-tool-iters', type=int, default=120)
    parser.add_argument('--max-tool-calls', type=int, default=400)
    parser.add_argument('--system-prompt-path', help='Explicit prompt file override.')
    parser.add_argument('--concurrency', type=int, default=8)
    parser.add_argument('--n-rollouts', type=int, default=4)
    parser.add_argument('--reward-metric', choices=('log_prob', 'brier'), default='brier')
    parser.add_argument('--format-coef', type=float, default=1.0)
    parser.add_argument('--no-trace', action='store_true')
    parser.add_argument('--run-id')
    early, _ = parser.parse_known_args(argv)
    if early.config:
        config = json.loads(early.config.read_text(encoding='utf-8'))
        known = {action.dest for action in parser._actions}
        unknown = set(config) - known
        if unknown:
            parser.error(f'Unknown config keys: {sorted(unknown)}')
        if isinstance(config.get('sampling_params'), dict):
            config['sampling_params'] = json.dumps(config['sampling_params'])
        parser.set_defaults(**config)
    args = parser.parse_args(argv)
    for action in parser._actions:
        value = getattr(args, action.dest, None)
        if action.choices and value is not None and value not in action.choices:
            parser.error(f'Invalid {action.dest}: {value!r}; choose from {action.choices}')
    args.no_tools = args.mode == 'no-tools'
    if args.base_url is None:
        args.base_url = {
            'anthropic': os.environ.get('ANTHROPIC_BASE_URL') or 'https://api.anthropic.com',
            'openai': os.environ.get('OPENAI_BASE_URL') or 'https://api.openai.com/v1',
            'openai-compatible': 'http://localhost:30000/v1',
            'sglang': 'http://localhost:30000',
        }[args.backend]
    if not args.base_url.strip() or any(not url.strip() for url in args.base_url.split(',')):
        parser.error('--base-url must contain nonempty endpoints')
    if args.backend != 'openai-compatible' and ',' in args.base_url:
        parser.error('Multiple endpoints are supported only by --backend openai-compatible')
    if args.no_thinking and args.backend != 'anthropic':
        parser.error('--no-thinking is supported only by --backend anthropic')
    if args.backend == 'anthropic' and args.reasoning_effort == 'none':
        parser.error('Use --no-thinking to disable Claude thinking')
    if not args.no_tools and not args.index_root:
        parser.error('Tool-enabled modes require --index-root or FORECAST_INDEX_ROOT')
    if args.backend != 'sglang' and not args.model_id:
        parser.error('--model-id is required for this backend')
    if (
        args.n_rollouts < 1 or args.concurrency < 1 or args.offset < 0
        or (args.limit is not None and args.limit < 1)
    ):
        parser.error('Rollouts, concurrency, and limit must be positive; offset must be nonnegative')
    if args.max_new_tokens is None:
        args.max_new_tokens = 65536 if args.no_tools else 32768
    if args.output is None:
        label = args.model_name or (args.model_id or 'model').rsplit('/', 1)[-1]
        args.output = f'results/{label}.{args.mode}.jsonl'
    return args


def main() -> None:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    asyncio.run(amain(parse_args()))


if __name__ == '__main__':
    main()
