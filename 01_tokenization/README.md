# Try These Experiments

## 1. Basic Token Count

```bash
python3 01_tokenization/tokenization_explorer.py --text "Hello world"
```

What to observe:

- Even short inputs consume multiple tokens.
- Token count is the base unit for both context usage and pricing.

## 2. Same Meaning, Different Length

```bash
python3 01_tokenization/tokenization_explorer.py --text "Summarize this article"
python3 01_tokenization/tokenization_explorer.py --text "Could you please provide a concise summary of the following article?"
```

What to observe:

- Longer phrasing usually increases token count.
- Prompt clarity should be balanced with token efficiency.

## 3. Compare Models

```bash
python3 01_tokenization/tokenization_explorer.py --text "Tokenization can vary by model." --model gpt-3.5-turbo
python3 01_tokenization/tokenization_explorer.py --text "Tokenization can vary by model." --model gpt-4
python3 01_tokenization/tokenization_explorer.py --text "Tokenization can vary by model." --model gpt-4o
```

What to observe:

- Tokenization may differ by model encoding.
- Cost estimates differ due to per-1k pricing defaults.

## 4. Estimate Cost with Custom Price

```bash
python3 01_tokenization/tokenization_explorer.py --text "A longer prompt to test pricing behavior and cost estimates." --price 0.01
```

What to observe:

- Cost scales linearly with token count: cost = (tokens / 1000) \* price.
- Custom pricing lets you simulate different providers or tiers.

## 5. Analyze a Real File

```bash
python3 01_tokenization/tokenization_explorer.py --file README.md --model gpt-4o
```

What to observe:

- Real documents consume far more tokens than short prompts.
- Useful for estimating prompt + context budget before API calls.

## 6. Inspect Raw Token IDs

```bash
python3 01_tokenization/tokenization_explorer.py --text "Tokenizer internals" --show-tokens
```

What to observe:

- If tiktoken is installed, you will see token IDs.
- If not installed, script falls back to a rough whitespace estimate.

## 7. Stress Test with Repetition

```bash
python3 01_tokenization/tokenization_explorer.py --text "test test test test test test test test test test"
python3 01_tokenization/tokenization_explorer.py --text "test1 test2 test3 test4 test5 test6 test7 test8 test9 test10"
```

What to observe:

- Similar-looking strings can map differently to tokens.
- Repetition patterns can affect tokenization efficiency.

## Quick Takeaways

- Token count drives both context usage and cost.
- Small wording changes can have measurable token impact.
- Always test representative real inputs, not just toy examples.
- Estimate cost early to avoid surprises in production.
