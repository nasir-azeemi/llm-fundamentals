"""Tokenization explorer utilities and simple CLI.

Features:
- Count tokens using `tiktoken` when available.
- Fallback to a simple whitespace-based estimator when `tiktoken` is not installed.
- Estimate cost given a price-per-1k-tokens value (configurable).

Usage examples:
python tokenization_explorer.py --text "Hello world" --model gpt-3.5-turbo
python tokenization_explorer.py --file notes.txt --model gpt-4 --price 0.03
"""
from __future__ import annotations

import argparse
import sys
from typing import Optional, Tuple

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except Exception:
    tiktoken = None  # type: ignore
    TIKTOKEN_AVAILABLE = False


DEFAULT_MODEL = "gpt-3.5-turbo"

# Example price-per-1k-tokens (USD). These are examples — replace with current prices.
EXAMPLE_PRICES_PER_1K = {
    "gpt-3.5-turbo": 0.002,
    "gpt-4": 0.03,
    "gpt-4o": 0.003,
}


def get_encoding_for_model(model: str):
    """Return a tiktoken encoding for a given model name when possible.

    Falls back to cl100k_base if model-specific encoding lookup fails.
    """
    if not TIKTOKEN_AVAILABLE:
        return None
    try:
        return tiktoken.encoding_for_model(model)
    except Exception:
        try:
            return tiktoken.get_encoding("cl100k_base")
        except Exception:
            return None


def count_tokens(text: str, model: str = DEFAULT_MODEL) -> Tuple[int, Optional[list]]:
    """Count tokens for `text`. Returns (num_tokens, tokens_list_or_None).

    If `tiktoken` is available, returns the actual token ids (list of ints).
    Otherwise, returns a rough whitespace-based estimate and None for token list.
    """
    if TIKTOKEN_AVAILABLE:
        enc = get_encoding_for_model(model)
        if enc is not None:
            token_ids = enc.encode(text)
            return len(token_ids), token_ids
    # Fallback naive estimator: split on whitespace and punctuation
    words = text.strip().split()
    return len(words), None


def estimate_cost(num_tokens: int, price_per_1k: float) -> float:
    """Estimate cost in the same currency as `price_per_1k`.

    Price is specified per 1000 tokens.
    """
    return (num_tokens / 1000.0) * price_per_1k


def format_currency(value: float) -> str:
    return f"${value:,.6f}" if value >= 0.001 else f"${value:.8f}"


def analyze_text(text: str, model: str = DEFAULT_MODEL, price_per_1k: Optional[float] = None):
    num_tokens, token_ids = count_tokens(text, model=model)
    price = price_per_1k if price_per_1k is not None else EXAMPLE_PRICES_PER_1K.get(
        model)
    cost = estimate_cost(num_tokens, price) if price is not None else None

    return {
        "model": model,
        "num_tokens": num_tokens,
        "token_ids": token_ids,
        "price_per_1k": price,
        "estimated_cost": cost,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Tokenization explorer: count tokens and estimate cost.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", help="Text to analyze")
    group.add_argument("--file", help="Path to a text file to analyze")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="Model name to choose tokenizer/price (default: %(default)s)")
    parser.add_argument("--price", type=float,
                        help="Custom price per 1000 tokens (USD). Overrides defaults.")
    parser.add_argument("--show-tokens", action="store_true",
                        help="Show raw token ids (if tiktoken available)")

    args = parser.parse_args(argv)

    if args.file:
        try:
            with open(args.file, "r", encoding="utf-8") as f:
                text = f.read()
        except Exception as e:
            print(f"Error reading file: {e}")
            sys.exit(2)
    else:
        text = args.text or ""

    result = analyze_text(text, model=args.model, price_per_1k=args.price)

    print(f"Model: {result['model']}")
    print(f"Tokens: {result['num_tokens']}")
    if args.show_tokens:
        if result["token_ids"] is not None:
            print("Token ids:")
            print(result["token_ids"])
        else:
            print("Token ids: (not available - tiktoken not installed)")

    if result["price_per_1k"] is not None:
        print(
            f"Price per 1k tokens: {format_currency(result['price_per_1k'])}")
        print(f"Estimated cost: {format_currency(result['estimated_cost'])}")
    else:
        print(
            "No price configured for this model. Pass --price to set a per-1k-token cost.")


if __name__ == "__main__":
    main()
