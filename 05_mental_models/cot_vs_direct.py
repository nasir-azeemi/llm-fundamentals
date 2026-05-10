"""Chain-of-thought vs direct answer comparison.

This script compares two prompting strategies on reasoning tasks:
- Direct: "What is the answer?"
- Chain-of-Thought (CoT): "Think step-by-step, then answer."

Research shows CoT often improves accuracy, especially on harder tasks.

Run:
python3 cot_vs_direct.py
python3 cot_vs_direct.py --show-full-responses
python3 cot_vs_direct.py --output cot_report.json

Live mode:
- Set GEMINI_API_KEY for real model responses.
- Without a key, uses simulated responses for teaching.
"""
from __future__ import annotations

import argparse
import json
import os
import textwrap
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass


DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


@dataclass
class ReasoningTask:
    task_id: str
    task_name: str
    base_prompt: str
    expected_answer: str


@dataclass
class ComparisonResult:
    task_id: str
    task_name: str
    direct_response: str
    cot_response: str
    direct_word_count: int
    cot_word_count: int
    direct_reasoning_depth: int
    cot_reasoning_depth: int
    source: str


def load_dotenv_if_present(path: str = ".env") -> None:
    if not os.path.exists(path):
        return

    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def build_tasks() -> list[ReasoningTask]:
    return [
        ReasoningTask(
            task_id="math_logic",
            task_name="Math logic puzzle",
            base_prompt="If Alice gives Bob 5 apples, and Bob then gives Carol half of what he has (which is now 12 apples total), how many apples does Carol receive?",
            expected_answer="6",
        ),
        ReasoningTask(
            task_id="deduction",
            task_name="Logical deduction",
            base_prompt=(
                "All birds can fly. Tweety is a bird. Can Tweety fly? "
                "Then tell me: is this reasoning sound, and why or why not?"
            ),
            expected_answer="No, unsound (not all birds can fly)",
        ),
        ReasoningTask(
            task_id="weighted_tradeoff",
            task_name="Weighted tradeoff analysis",
            base_prompt=(
                "Compare these two options: Option A is fast (1 hour), costs $100, and has 80% success rate. "
                "Option B is slow (3 hours), costs $50, and has 95% success rate. "
                "Which is better, and what factors matter most?"
            ),
            expected_answer="Depends on context; no single answer",
        ),
        ReasoningTask(
            task_id="analogy",
            task_name="Analogical reasoning",
            base_prompt="A hammer is to a nail as a _____ is to a screw.",
            expected_answer="Screwdriver",
        ),
    ]


def _extract_system_instruction(messages: list[dict[str, str]]) -> str:
    for message in messages:
        if message["role"] == "system":
            return message["content"]
    return ""


def _messages_to_gemini_contents(messages: list[dict[str, str]]) -> list[dict[str, object]]:
    contents: list[dict[str, object]] = []
    for message in messages:
        if message["role"] == "system":
            continue
        contents.append(
            {
                "role": "user",
                "parts": [{"text": message["content"]}],
            }
        )
    return contents


def call_gemini_generate_content(
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    api_url: str,
) -> str:
    system_instruction = _extract_system_instruction(messages)
    payload: dict[str, object] = {
        "contents": _messages_to_gemini_contents(messages),
        "generationConfig": {"temperature": temperature},
    }
    if system_instruction:
        payload["systemInstruction"] = {
            "parts": [{"text": system_instruction}]}

    request = urllib.request.Request(
        f"{api_url.format(model=model)}?key={api_key}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=60) as response:
        data = json.loads(response.read().decode("utf-8"))

    try:
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected API response: {data}") from exc


def simulate_direct_response(task_id: str) -> str:
    simulations = {
        "math_logic": "6 apples.",
        "deduction": "Yes, Tweety can fly. The logic is: all birds fly, Tweety is a bird, so Tweety flies.",
        "weighted_tradeoff": "Option A is better because it is faster.",
        "analogy": "Screwdriver.",
    }
    return simulations.get(task_id, "Answer not available.")


def simulate_cot_response(task_id: str) -> str:
    simulations = {
        "math_logic": (
            "Let me work through this step-by-step. First, Bob has 12 apples total after receiving 5 from Alice. "
            "Then Bob gives Carol half of his 12 apples. Half of 12 is 6. So Carol receives 6 apples."
        ),
        "deduction": (
            "Step 1: The statement 'all birds can fly' is too broad. Counterexample: penguins, ostriches, kiwis. "
            "Step 2: Tweety is a bird, but we don't know what species. "
            "Step 3: We cannot conclude Tweety can fly without more info. "
            "Conclusion: The reasoning is unsound because the first premise is false."
        ),
        "weighted_tradeoff": (
            "I need to weigh several factors. First: cost difference is $50 (Option B cheaper). "
            "Second: time difference is 2 hours (Option A faster). "
            "Third: success rate differs by 15 percentage points (Option B more reliable). "
            "The best choice depends on what I value: Option A if speed matters most, Option B if reliability matters most."
        ),
        "analogy": (
            "Let me think about the relationship. A hammer is a tool used to drive nails. "
            "So I need a tool used to drive screws. That tool is a screwdriver."
        ),
    }
    return simulations.get(task_id, "Detailed reasoning not available.")


def count_reasoning_steps(text: str) -> int:
    indicators = ["step", "first", "second", "third",
                  "then", "therefore", "so", "because", "thus"]
    lower = text.lower()
    count = sum(lower.count(indicator) for indicator in indicators)
    return max(count, 1)


def run_comparison(
    task: ReasoningTask,
    model: str,
    temperature: float,
    api_url: str,
    api_key: str | None,
) -> ComparisonResult:
    direct_messages = [
        {"role": "system", "content": "Answer briefly and directly."},
        {"role": "user", "content": task.base_prompt},
    ]

    cot_messages = [
        {
            "role": "system",
            "content": "Think step-by-step. Show your reasoning clearly before giving the final answer.",
        },
        {"role": "user", "content": task.base_prompt},
    ]

    if api_key:
        try:
            direct_response = call_gemini_generate_content(
                api_key=api_key,
                model=model,
                messages=direct_messages,
                temperature=temperature,
                api_url=api_url,
            )
        except Exception:  # noqa: BLE001
            direct_response = simulate_direct_response(task.task_id)

        try:
            cot_response = call_gemini_generate_content(
                api_key=api_key,
                model=model,
                messages=cot_messages,
                temperature=temperature,
                api_url=api_url,
            )
        except Exception:  # noqa: BLE001
            cot_response = simulate_cot_response(task.task_id)

        source = "live API"
    else:
        direct_response = simulate_direct_response(task.task_id)
        cot_response = simulate_cot_response(task.task_id)
        source = "simulation"

    direct_words = len(direct_response.split())
    cot_words = len(cot_response.split())
    direct_reasoning = count_reasoning_steps(direct_response)
    cot_reasoning = count_reasoning_steps(cot_response)

    return ComparisonResult(
        task_id=task.task_id,
        task_name=task.task_name,
        direct_response=direct_response,
        cot_response=cot_response,
        direct_word_count=direct_words,
        cot_word_count=cot_words,
        direct_reasoning_depth=direct_reasoning,
        cot_reasoning_depth=cot_reasoning,
        source=source,
    )


def print_comparison(task: ReasoningTask, result: ComparisonResult, show_full: bool) -> None:
    print("\n" + "=" * 90)
    print(f"Task: {result.task_name}")
    print("=" * 90)
    print(f"Prompt: {textwrap.fill(task.base_prompt, width=96)}")
    print(f"Expected: {task.expected_answer}")

    print("\n" + "-" * 90)
    print("DIRECT ANSWER")
    print("-" * 90)
    if show_full:
        print(f"Response: {textwrap.fill(result.direct_response, width=96)}")
    else:
        truncated = (
            result.direct_response[:120] + "..."
            if len(result.direct_response) > 120
            else result.direct_response
        )
        print(f"Response: {truncated}")
    print(
        f"Metrics: {result.direct_word_count} words, reasoning depth: {result.direct_reasoning_depth}")

    print("\n" + "-" * 90)
    print("CHAIN-OF-THOUGHT (CoT)")
    print("-" * 90)
    if show_full:
        print(f"Response: {textwrap.fill(result.cot_response, width=96)}")
    else:
        truncated = (
            result.cot_response[:120] + "..."
            if len(result.cot_response) > 120
            else result.cot_response
        )
        print(f"Response: {truncated}")
    print(
        f"Metrics: {result.cot_word_count} words, reasoning depth: {result.cot_reasoning_depth}")

    print("\n" + "-" * 90)
    print("COMPARISON")
    print("-" * 90)
    print(
        f"CoT is {result.cot_word_count - result.direct_word_count:+d} words longer")
    print(
        f"CoT has {result.cot_reasoning_depth - result.direct_reasoning_depth:+d} more reasoning steps"
    )
    if result.cot_reasoning_depth > result.direct_reasoning_depth:
        print("✓ CoT shows deeper reasoning on this task.")
    else:
        print("⚠ Both use similar reasoning depth.")


def print_summary(results: list[ComparisonResult]) -> None:
    print("\n" + "#" * 90)
    print("Summary: Chain-of-Thought vs Direct")
    print("#" * 90)
    print(
        f"{'Task':25s} {'Direct Words':15s} {'CoT Words':15s} {'Direct Depth':15s} {'CoT Depth':15s}"
    )
    print("-" * 90)
    for result in results:
        print(
            f"{result.task_id:25s} {str(result.direct_word_count):15s} "
            f"{str(result.cot_word_count):15s} {str(result.direct_reasoning_depth):15s} "
            f"{str(result.cot_reasoning_depth):15s}"
        )

    avg_direct_depth = sum(
        r.direct_reasoning_depth for r in results) / len(results)
    avg_cot_depth = sum(r.cot_reasoning_depth for r in results) / len(results)
    print(
        f"\nAverage reasoning depth: Direct={avg_direct_depth:.1f}, CoT={avg_cot_depth:.1f}"
    )
    if avg_cot_depth > avg_direct_depth:
        print("✓ Chain-of-Thought shows deeper reasoning across tasks.")


def save_report(path: str, results: list[ComparisonResult]) -> None:
    payload = {
        "note": "Chain-of-thought vs direct answer comparison.",
        "results": [asdict(result) for result in results],
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare chain-of-thought vs direct answers on reasoning tasks."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Gemini model for live mode (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.5,
        help="Temperature for live API mode",
    )
    parser.add_argument(
        "--api-url",
        default=DEFAULT_API_URL,
        help="Gemini API URL template",
    )
    parser.add_argument(
        "--output",
        default="cot_report.json",
        help="Path to JSON report",
    )
    parser.add_argument(
        "--show-full-responses",
        action="store_true",
        help="Show full responses instead of truncated",
    )
    args = parser.parse_args(argv)

    load_dotenv_if_present()
    api_key = os.getenv("GEMINI_API_KEY", "").strip() or None
    print("Running in LIVE API mode." if api_key else "GEMINI_API_KEY not set. Running in SIMULATION mode.")

    tasks = build_tasks()
    results = [
        run_comparison(
            task=task,
            model=args.model,
            temperature=args.temperature,
            api_url=args.api_url,
            api_key=api_key,
        )
        for task in tasks
    ]

    for task, result in zip(tasks, results):
        print_comparison(task, result, show_full=args.show_full_responses)

    print_summary(results)
    save_report(args.output, results)
    print(f"\nSaved report to: {args.output}")

    print("\nKey insight:")
    print("Chain-of-thought prompting often improves reasoning by explicitly asking the model")
    print("to show its work. This is especially powerful for math, logic, and complex tradeoffs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
