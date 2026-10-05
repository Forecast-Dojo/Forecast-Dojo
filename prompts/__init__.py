"""System prompts for forecasting agents."""

from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parent


def prompt_filename(mode: str, *, search: bool = True, enable_code: bool = True) -> str:
    names = {"memory-free": "memory-free", "memory-on": "memory-on"}
    if not search or mode == "no-tools":
        return "no-tools.md"
    name = names[mode]
    suffix = "" if enable_code else "-no-code"
    return f"{name}{suffix}.md"


def prompt_path(mode: str, *, search: bool = True, enable_code: bool = True) -> Path:
    return PROMPTS_DIR / prompt_filename(mode, search=search, enable_code=enable_code)


def load_prompt(mode: str, *, search: bool = True, enable_code: bool = True) -> str:
    return prompt_path(mode, search=search, enable_code=enable_code).read_text(encoding="utf-8").strip()
