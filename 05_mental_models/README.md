# Try These Experiments

## 1. Hallucination Risk Scoring (5 risk patterns)

```bash
python3 hallucination_demo.py
# Output: hallucination_report.json with risk scores for each pattern

# Run with full responses:
python3 hallucination_demo.py --show-full-responses

# Save to custom location:
python3 hallucination_demo.py --output hallucination_analysis.json
```

**What to look for**: Risk scores of 6+ (high risk), 3–5 (medium), <3 (low). Compare risk signals across cases.

## 2. Chain-of-Thought vs Direct (4 reasoning tasks)

```bash
python3 cot_vs_direct.py
# Output: cot_report.json showing word count and reasoning depth

# Show full responses (not truncated):
python3 cot_vs_direct.py --show-full-responses

# Use a different model:
python3 cot_vs_direct.py --model gemini-1.5-flash

# Adjust reasoning temperature (0.5 = moderate):
python3 cot_vs_direct.py --temperature 0.3
```

**What to look for**:

- CoT typically uses 5–10x more tokens
- Reasoning depth (step count) increases substantially for complex tasks
- Average reasoning depth: Direct ≈ 2, CoT ≈ 6+

## 3. Hallucination Risk Under Different Temperatures

```bash
# Lower temperature = more deterministic, often fewer hallucinations
python3 hallucination_demo.py
GEMINI_API_KEY="YOUR_KEY" python3 hallucination_demo.py --temperature 0.2

# Higher temperature = more creative but higher hallucination risk
GEMINI_API_KEY="YOUR_KEY" python3 hallucination_demo.py --temperature 1.0

# Compare the risk scores across runs
```

## 4. CoT on Your Own Problem

Modify `cot_vs_direct.py` to add your own reasoning task:

```python
ReasoningTask(
    task_id="your_task",
    task_name="Your reasoning challenge",
    base_prompt="Your problem statement here",
    expected_answer="What you expect to see",
)
```

Then run:

```bash
python3 cot_vs_direct.py --show-full-responses
```

---

# Best Practices

## For Reducing Hallucinations

1. **Ask for uncertainty**: "Rate your confidence (0–100%) for each claim"
2. **Cite sources**: "Provide a URL or specific source for this information"
3. **Verify APIs**: Always check against official docs before using generated code
4. **Use lower temperature** for fact-heavy tasks (0.2–0.5)
5. **Add verification step**: "How do you know this is true?"

## For Leveraging CoT

1. **Use on hard tasks** (math, logic, complex tradeoffs)
2. **Skip on simple tasks** (lookup, analogy when obvious)
3. **Add examples** for specialized domains (few-shot CoT)
4. **Verify intermediate steps**, not just the final answer
5. **Combine with lower temperature** for reliable reasoning
