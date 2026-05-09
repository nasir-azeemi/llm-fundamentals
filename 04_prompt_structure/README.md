# Learnings

Same instruction can lead to different behavior depending on whether it is placed in a system message or a user message.

System instructions act like higher-priority guidance, so they are more stable when the user later adds a conflicting request.

User instructions are easier to override when the conversation contains stronger role-level guidance.

Personas shape the model's tone, depth, and style through a system message that defines a role such (eg. teacher, skeptic, executive, engineer, or coach).

Different personas can produce different answers from the same prompt, even when the underlying task stays the same.

System-vs-user experiments are a good way to see instruction hierarchy in practice because they make conflicts visible.

Persona experiments are a good way to see how style, framing, and priorities change without changing the core user question.

Put the most important rules in the system message and keep the user message focused on the task.

To explore output variety, compare the same prompt across multiple personas and observe how the framing changes.

## Try These Experiments

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
