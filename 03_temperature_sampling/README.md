# Learnings

Temperature controls the randomness and creativity of an LLM's output during token sampling.

Temperature is applied to the model's output logits before converting them to probabilities via softmax. (Remember a LLM is a text completion engine)

Temperature = 0: Deterministic sampling (always picks the token with the highest probability).

Temperature = 1.0: Standard softmax behavior (neutral baseline, no scaling applied).

Temperature < 1.0: Makes the probability distribution sharper/more peaked (less randomness, more repetitive).

Temperature > 1.0: Makes the probability distribution flatter/more uniform (more randomness, more creative).

Low temperature (0-0.3) is good for tasks requiring consistency and factual accuracy like question-answering or code generation.

Medium temperature (0.5-0.8) balances consistency with some creativity, useful for general conversation.

High temperature (1.0-2.0) increases diversity and creativity but may produce less coherent or factually incorrect outputs.

Entropy measures the randomness of the probability distribution. Higher entropy means the model is more likely to sample from a wider variety of tokens.

Different models may have different recommended temperature ranges for best results.

Top-k sampling and top-p (nucleus) sampling are often used alongside temperature to limit the tokens the model can choose from, preventing very unlikely tokens.

You can adjust temperature based on the task: lower for precise outputs, higher for creative/diverse outputs.

## Try These Experiments

Run the `temperature_comparison.py` script to see how temperature affects token probabilities:

**Compare deterministic vs creative sampling:**

```
python3 temperature_comparison.py --prompt "The future of AI is" --temps 0 0.5 1.0 1.5
```

**Focus on low temperatures (deterministic range):**

```
python3 temperature_comparison.py --prompt "2 + 2 =" --temps 0 0.1 0.3 0.5
```

**Focus on high temperatures (creative range):**

```
python3 temperature_comparison.py --prompt "Write a story beginning with" --temps 0.8 1.0 1.5 2.0
```

**Custom prompt with specific temperatures:**

```
python3 temperature_comparison.py --prompt "Your custom prompt here" --temps 0.3 0.7 1.2
```

Look at how the entropy changes and how the probability distribution flattens as temperature increases.
