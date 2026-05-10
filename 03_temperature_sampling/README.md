# Try These Experiments

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
