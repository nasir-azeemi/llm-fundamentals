# Learnings

A helpful rule of thumb is that one token generally corresponds to ~4 characters of text for common English text. This translates to roughly ¾ of a word (so 100 tokens ~= 75 words).

In any given LLM, tokens are mapped to integers, where a single token ID represents one specific, unique character, word, or subword.

The tokenizer converts text into a sequence of these unique IDs. This is called encoding.

Token IDs are not universal. A token ID for "cat" in one model may be entirely different in another model.

The same tokenizer will map the same token string to the same numerical ID every time to ensure consistent processing.

Tokens are the currency of LLMs. Total cost of an api call is the sum of cost for input tokens and output tokens.

Token vocabulary size is important. The bigger the vocabulary the fewer token are needed to encode a sentence. Fewer tokens mean more efficient processing by LLMs but also means they need to be larger and require more memory to execution.

If a word in unknown to the LLM it is broken down into more tokens. That means a LLM trained on english language will not perform efficiently on the same prompt in another language.
