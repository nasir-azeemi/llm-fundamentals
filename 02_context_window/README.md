# Try These Experiments

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
