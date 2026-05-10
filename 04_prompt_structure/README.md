# Try These Experiments

Run the system-vs-user comparison:

```
python3 system_vs_user_prompt.py
```

Run the persona comparison:

```
python3 role_prompting.py
```

Try a stricter instruction with a conflicting follow-up:

```
python3 system_vs_user_prompt.py --instruction "Answer in exactly 5 words." --query "What is overfitting?"
```

Try the same prompt across different personas:

```
python3 role_prompting.py --prompt "Explain what makes a prompt effective." --personas teacher skeptic executive engineer coach
```
