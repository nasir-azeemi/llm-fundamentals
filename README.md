On my journey to learn AI engineering. This is just a small compilation of my learnings regarding LLMs. It has helped me understand the basics and now I am more conscious of how I write prompts.

# LLM-fundamentals

An LLM is a probabilistic text completion engine with a massive compressed representation of human knowledge. It is not a database, not a search engine, not a reasoning engine — though it can simulate all of those.

# Learnings

## Tokenization

A helpful rule of thumb is that one token generally corresponds to ~4 characters of text for common English text. This translates to roughly ¾ of a word (so 100 tokens ~= 75 words).

In any given LLM, tokens are mapped to integers, where a single token ID represents one specific, unique character, word, or subword.

The tokenizer converts text into a sequence of these unique IDs. This is called encoding.

Token IDs are not universal. A token ID for "cat" in one model may be entirely different in another model.

The same tokenizer will map the same token string to the same numerical ID every time to ensure consistent processing.

Tokens are the currency of LLMs. Total cost of an api call is the sum of cost for input tokens and output tokens.

Token vocabulary size is important. The bigger the vocabulary the fewer token are needed to encode a sentence. Fewer tokens mean more efficient processing by LLMs but also means they need to be larger and require more memory to execution.

If a word in unknown to the LLM it is broken down into more tokens. That means a LLM trained on english language will not perform efficiently on the same prompt in another language.

## Context Window

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

## Temperature

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

## Prompting

Same instruction can lead to different behavior depending on whether it is placed in a system message or a user message.

System instructions act like higher-priority guidance, so they are more stable when the user later adds a conflicting request.

User instructions are easier to override when the conversation contains stronger role-level guidance.

Personas shape the model's tone, depth, and style through a system message that defines a role such (eg. teacher, skeptic, executive, engineer, or coach).

Different personas can produce different answers from the same prompt, even when the underlying task stays the same.

System-vs-user experiments are a good way to see instruction hierarchy in practice because they make conflicts visible.

Persona experiments are a good way to see how style, framing, and priorities change without changing the core user question.

Put the most important rules in the system message and keep the user message focused on the task.

To explore output variety, compare the same prompt across multiple personas and observe how the framing changes.

## Hallucination

Hallucinations occur when an LLM generates confident false information. The model doesn't "know" it's wrong—it outputs plausible-sounding text that doesn't match reality.

Happens because LLMs optimize for _fluent, plausible-sounding text_, not accuracy. If training data contains gaps; model fills them statistically, not factually. LLms usually don't say "I don't know" when it should.

### Some Hallucination Patterns

1. **Fabricated Citations**: Inventing fake papers, DOIs, or academic sources
2. **Fictional People Confabulation**: Creating detailed biographies of non-existent researchers or public figures
3. **Non-existent APIs/Methods**: Inventing functions or methods that don't exist in libraries
4. **Future Facts**: Stating unknowable future events as definite facts
5. **Quote Fabrication**: Creating verbatim quotes from non-existent or unread sources

_Although in my experiments only Fictional entity confabulation worked. LLMs are much better now compared to last year. A better test would be to check the same prompts across multiple models_

To avoid hallucination ask the LLM to cite specific documents or URLs. Lower temps (0.3–0.5) reduce hallucination (but also reduce creativity). Explicitly prompt: "State your confidence level (0–100%) before each factual claim" or add a secondary prompt asking the model explain/critique its output.

## Chain-of-thought (CoT) vs direct reasoning

CoT prompting asks a model to **show its reasoning step-by-step** before reaching a conclusion. Instead of "What is the answer?", ask "Let's think step-by-step: ...".

Cot allows LLMs to go through intermediate step. Wrong steps become visible early, allowing self-correction. Breaking any problem into steps makes hard tasks more tractable.

CoT does improve accuracy especially for hard tasks but uses more tokens, increasing cost and latency.

|                      | Connection To Previous Topics                                                                     |
| -------------------- | ------------------------------------------------------------------------------------------------- |
| **Tokenization**     | CoT uses 5–10x more tokens; hallucinations cost you verification tokens                           |
| **Context Windows**  | Long CoT chains can approach context limits; hallucinations fill unused tokens                    |
| **Temperature**      | Lower temp reduces hallucinations; higher temp enables better exploratory reasoning (if accurate) |
| **Prompt Structure** | System instructions can reduce hallucinations (grounding); personas affect reasoning style        |

---
