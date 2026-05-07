"""Context window experiment.

This script helps you understand:
- How many tokens a prompt uses
- How close that prompt is to a model's context window
- What happens when you approach the limit
- What happens when you exceed the limit

Examples:
python context_limit_experiment.py --text "Write a haiku about token limits" --window-size 128
python context_limit_experiment.py --file notes.txt --model gpt-4o --reserve-output 200
python context_limit_experiment.py --text "..." --strategy truncate
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    import tiktoken
except Exception:  # pragma: no cover - fallback is intentional
    tiktoken = None


DEFAULT_MODEL = "gpt-4o"
DEFAULT_WINDOW_SIZE = 128000
DEFAULT_RESERVE_OUTPUT = 1000


@dataclass
class ContextReport:
    model: str
    window_size: int
    reserve_output_tokens: int
    prompt_tokens: int
    prompt_token_ids: Optional[list[int]]
    utilization_pct: float
    remaining_tokens_for_prompt_and_output: int
    max_prompt_tokens_if_reserving_output: int
    status: str
    behavior: str
    truncated_text: Optional[str] = None
    truncated_tokens: Optional[int] = None


def get_encoding(model: str):
    if tiktoken is None:
        return None

    try:
        return tiktoken.encoding_for_model(model)
    except Exception:
        return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str, model: str) -> tuple[int, Optional[list[int]]]:
    encoding = get_encoding(model)
    if encoding is None:
        words = text.split()
        return len(words), None

    token_ids = encoding.encode(text)
    return len(token_ids), token_ids


def truncate_to_token_budget(text: str, model: str, token_budget: int) -> tuple[str, int]:
    encoding = get_encoding(model)
    if encoding is None:
        words = text.split()
        truncated_words = words[: max(token_budget, 0)]
        truncated_text = " ".join(truncated_words)
        return truncated_text, len(truncated_words)

    token_ids = encoding.encode(text)
    kept_ids = token_ids[: max(token_budget, 0)]
    truncated_text = encoding.decode(kept_ids)
    return truncated_text, len(kept_ids)


def classify_utilization(utilization_pct: float) -> str:
    if utilization_pct < 70:
        return "safe"
    if utilization_pct < 90:
        return "approaching_limit"
    if utilization_pct <= 100:
        return "near_limit"
    return "over_limit"


def analyze_context_window(
    text: str,
    model: str = DEFAULT_MODEL,
    window_size: int = DEFAULT_WINDOW_SIZE,
    reserve_output_tokens: int = DEFAULT_RESERVE_OUTPUT,
    strategy: str = "reject",
) -> ContextReport:
    prompt_tokens, token_ids = count_tokens(text, model)
    total_budget = window_size
    remaining = total_budget - prompt_tokens - reserve_output_tokens
    max_prompt_tokens = max(window_size - reserve_output_tokens, 0)
    utilization_pct = (prompt_tokens / window_size) * 100 if window_size else 0.0
    status = classify_utilization(utilization_pct)

    behavior: str
    truncated_text: Optional[str] = None
    truncated_tokens: Optional[int] = None

    if remaining >= 0:
        behavior = (
            "The prompt fits. The model can process the input and still has room "
            "for the reserved output tokens."
        )
    elif strategy == "truncate":
        keep_budget = max_prompt_tokens
        truncated_text, truncated_tokens = truncate_to_token_budget(text, model, keep_budget)
        behavior = (
            "The prompt exceeds the window, so this demo truncates the input to fit. "
            "Anything beyond the budget is dropped before the model sees it."
        )
    else:
        behavior = (
            "The prompt exceeds the window. In a real API call, this usually means the request "
            "is rejected or the application must shorten the input first."
        )

    return ContextReport(
        model=model,
        window_size=window_size,
        reserve_output_tokens=reserve_output_tokens,
        prompt_tokens=prompt_tokens,
        prompt_token_ids=token_ids,
        utilization_pct=utilization_pct,
        remaining_tokens_for_prompt_and_output=remaining,
        max_prompt_tokens_if_reserving_output=max_prompt_tokens,
        status=status,
        behavior=behavior,
        truncated_text=truncated_text,
        truncated_tokens=truncated_tokens,
    )


def print_report(report: ContextReport, show_tokens: bool = False) -> None:
    print("Context Window Report")
    print("-" * 22)
    print(f"Model: {report.model}")
    print(f"Window size: {report.window_size} tokens")
    print(f"Reserved for output: {report.reserve_output_tokens} tokens")
    print(f"Prompt tokens: {report.prompt_tokens}")
    print(f"Utilization: {report.utilization_pct:.2f}%")
    print(f"Status: {report.status}")
    print(f"Remaining tokens after reserving output: {report.remaining_tokens_for_prompt_and_output}")
    print(f"Max prompt tokens if reserving output: {report.max_prompt_tokens_if_reserving_output}")
    print()
    print("What happens:")
    print(report.behavior)

    if report.remaining_tokens_for_prompt_and_output < 0:
        over_by = abs(report.remaining_tokens_for_prompt_and_output)
        print()
        print(f"Over limit by: {over_by} tokens")

    if report.truncated_text is not None:
        print()
        print("Truncation demo:")
        print(f"Kept tokens: {report.truncated_tokens}")
        print("Truncated text:")
        print(report.truncated_text)

    if show_tokens and report.prompt_token_ids is not None:
        print()
        print("Token ids:")
        print(report.prompt_token_ids)


def read_input_text(text: Optional[str], file_path: Optional[str]) -> str:
    if file_path:
        return Path(file_path).read_text(encoding="utf-8")
    return text or ""


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Explore LLM context windows and token limits.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", help="Text to analyze")
    source.add_argument("--file", help="Path to a text file to analyze")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Model name for tokenization (default: {DEFAULT_MODEL})")
    parser.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE, help=f"Context window size in tokens (default: {DEFAULT_WINDOW_SIZE})")
    parser.add_argument("--reserve-output", type=int, default=DEFAULT_RESERVE_OUTPUT, help=f"Tokens to reserve for the model response (default: {DEFAULT_RESERVE_OUTPUT})")
    parser.add_argument(
        "--strategy",
        choices=("reject", "truncate"),
        default="reject",
        help="Behavior to demonstrate when the prompt exceeds the limit",
    )
    parser.add_argument("--show-tokens", action="store_true", help="Print token ids when available")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    text = read_input_text(args.text, args.file)
    report = analyze_context_window(
        text=text,
        model=args.model,
        window_size=args.window_size,
        reserve_output_tokens=args.reserve_output,
        strategy=args.strategy,
    )
    print_report(report, show_tokens=args.show_tokens)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())