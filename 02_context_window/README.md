# Learnings

A context window is the maximum amount of tokens a LLM can consider at once.

The prompt (input) and the response (output) both make up the context window, so leaving a reserve for output is important.

The total number of token grow as a the conversation gets longer.

As a prompt gets closer to the limit (which is set by the LLM provider), the LLM has less room for additional input and the response.

If the input exceeds the limit, the application must either reject the request (context window limit hit error) or remove some text before sending it.

Truncation can change the meaning of a prompt because the LLM never sees the dropped tokens.

Different LLMs can have different context window sizes, so the same prompt may fit in one LLM and fail in another.

Long documents often need to be chunked or summarized before they can be sent to an LLM.

LLM processing is expensive and so adding more text and more context window means you're using more memory per process.

Larger the context window the more performance degrades because all models (small or large) suffer from a problem of retrieving information from their own context.

Stuff at the start and the stuff at the end is deemed most important by the attention mechanism that the LLM uses. This is an emergent property of how these systems are designed. Kinda like how humans have primacy bias and recency bias.

Models do better with less and more focused information.

MCP servers are attractive because they allow you to plug and play with different pre-made tool sets but they can bloat your context quickly.

## Try These Experiments

Run the `context_limit_experiment.py` script to see context windows in action:

**Safe prompt (well within the window):**

```
python3 context_limit_experiment.py --text "Hello world" --window-size 100
```

**Approaching the limit:**

```
python3 context_limit_experiment.py --text "Write a detailed explanation of machine learning" --window-size 50
```

**Exceeding the limit with truncation:**

```
python3 context_limit_experiment.py --text "Lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor incididunt ut labore et dolore magna aliqua" --window-size 20 --reserve-output 5 --strategy truncate
```

**Compare models with different context windows:**

```
python3 context_limit_experiment.py --text "Your text here" --model gpt-4 --window-size 8000
python3 context_limit_experiment.py --text "Your text here" --model gpt-4o --window-size 128000
```

**See token IDs (when tiktoken is installed):**

```
python3 context_limit_experiment.py --text "Short text" --show-tokens
```
