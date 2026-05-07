"""Temperature comparison and sampling exploration.

This script helps you understand how temperature affects LLM output:
- Temperature 0: Deterministic (always picks highest probability token)
- Temperature 0.5-1.0: Balanced creativity and consistency
- Temperature > 1.0: More creative and random

Features:
- Shows how temperature transforms probability distributions
- Simulates token sampling at different temperatures
- Can compare multiple temperature values side-by-side

Usage:
python3 temperature_comparison.py --prompt "Complete this: The future of AI is" --temps 0 0.5 1.0 1.5
python3 temperature_comparison.py --prompt "..." --model gpt-4o --temps 0 0.7
"""
from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from typing import Optional


@dataclass
class TokenProb:
    token: str
    logit: float
    base_prob: float


@dataclass
class TemperatureResult:
    temperature: float
    probabilities: dict[str, float]
    top_tokens: list[tuple[str, float]]
    selected_token: str
    entropy: float


def apply_temperature(logits: list[float], temperature: float) -> list[float]:
    """Apply temperature scaling to logits.

    Temperature > 1.0 makes distribution more uniform (more random).
    Temperature < 1.0 makes distribution more peaked (more deterministic).
    Temperature = 1.0 is neutral.
    Temperature = 0.0 is deterministic (argmax).
    """
    if temperature < 0:
        raise ValueError("Temperature must be non-negative")

    # Handle temperature 0 case (deterministic: always pick top)
    if temperature == 0:
        max_logit = max(logits)
        probs = [1.0 if logit == max_logit else 0.0 for logit in logits]
        # Handle ties by distributing equally
        num_max = sum(1 for logit in logits if logit == max_logit)
        if num_max > 1:
            prob_value = 1.0 / num_max
            probs = [prob_value if logit == max_logit else 0.0 for logit in logits]
        return probs

    # Scale logits by temperature
    scaled_logits = [logit / temperature for logit in logits]

    # Compute softmax for probabilities
    max_logit = max(scaled_logits)
    exp_logits = [math.exp(logit - max_logit) for logit in scaled_logits]
    sum_exp = sum(exp_logits)
    probs = [exp_logit / sum_exp for exp_logit in exp_logits]

    return probs


def compute_entropy(probs: list[float]) -> float:
    """Compute Shannon entropy of a probability distribution."""
    entropy = 0.0
    for p in probs:
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def sample_token(probs: list[float], tokens: list[str]) -> str:
    """Sample a token according to the probability distribution using standard library."""
    cumsum = 0.0
    rand = random.random()
    for token, prob in zip(tokens, probs):
        cumsum += prob
        if rand < cumsum:
            return token
    return tokens[-1]  # fallback


def analyze_temperature(
    token_labels: list[str],
    logits: list[float],
    temperature: float,
) -> TemperatureResult:
    """Analyze how a given temperature affects sampling from logits."""
    probs = apply_temperature(logits, temperature)

    # Build probability dict
    prob_dict = {token: prob for token, prob in zip(token_labels, probs)}

    # Get top 5 tokens
    sorted_tokens = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
    top_tokens = sorted_tokens[:5]

    # Sample a token
    sampled_token = sample_token(probs, token_labels)

    # Compute entropy
    entropy = compute_entropy(probs)

    return TemperatureResult(
        temperature=temperature,
        probabilities=prob_dict,
        top_tokens=top_tokens,
        selected_token=sampled_token,
        entropy=entropy,
    )


def print_temperature_report(result: TemperatureResult) -> None:
    """Pretty-print a temperature analysis result."""
    print(f"\n{'='*60}")
    print(f"Temperature: {result.temperature}")
    print(f"{'='*60}")
    print(f"Entropy: {result.entropy:.4f} (higher = more randomness)")
    print(f"Sampled token: {result.selected_token}")
    print("\nTop 5 tokens by probability:")
    for token, prob in result.top_tokens:
        bar_length = int(prob * 50)
        bar = "█" * bar_length
        print(f"  {token:15s} {prob:6.2%}  {bar}")


def compare_temperatures(
    prompt: str,
    temperatures: list[float],
    num_samples: int = 1,
) -> None:
    """Compare outputs at multiple temperatures.

    For demo purposes, we use simulated logits.
    In a real scenario, these would come from a model's final layer.
    """
    print(f"Prompt: {prompt}")
    print(f"Temperatures: {temperatures}")
    print(f"Samples per temperature: {num_samples}")

    # Simulated token vocabulary and logits (pretend these come from a model)
    token_labels = [
        "innovation",
        "creativity",
        "uncertainty",
        "advancement",
        "challenge",
        "opportunity",
        "risk",
        "unknown",
        "potential",
        "transformation",
    ]
    logits = [4.2, 3.8, 2.1, 3.5, 1.8, 2.9, 1.2, 0.8, 3.1, 2.5]

    results = []
    for temp in temperatures:
        result = analyze_temperature(token_labels, logits, temp)
        results.append(result)
        print_temperature_report(result)

    # Summary comparison
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"{'Temp':<8} {'Top Token':<15} {'Top Prob':<10} {'Entropy':<10}")
    print("-" * 43)
    for result in results:
        top_token, top_prob = result.top_tokens[0]
        print(
            f"{result.temperature:<8.1f} {top_token:<15} {top_prob:<10.2%} {result.entropy:<10.4f}"
        )

    print("\nKey observations:")
    print("- Temp 0.0: Always picks the highest probability token (deterministic)")
    print("- Temp 0.5-1.0: Balanced between consistency and diversity")
    print("- Temp 1.0: Standard softmax (neutral baseline)")
    print("- Temp > 1.0: Flatter distribution, more randomness")
    print("- Higher entropy = more diverse sampling")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Explore temperature effects on LLM sampling.")
    parser.add_argument("--prompt", default="Complete this:", help="Prompt to analyze")
    parser.add_argument(
        "--temps",
        type=float,
        nargs="+",
        default=[0.0, 0.5, 1.0, 1.5],
        help="Temperature values to compare (default: 0.0 0.5 1.0 1.5)",
    )
    parser.add_argument("--samples", type=int, default=1, help="Number of samples per temperature")

    args = parser.parse_args(argv)

    compare_temperatures(
        prompt=args.prompt,
        temperatures=args.temps,
        num_samples=args.samples,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
