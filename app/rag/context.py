# app/rag/context.py
from app.models.llm import LLMModel

def truncate_context_by_tokens(texts, max_tokens=3000):
    # Get tokenizer from existing (or create-once) model
    tokenizer = LLMModel().tokenizer  # ← safe: returns existing instance

    context_parts = []
    total_tokens = 0
    for text in texts:
        tokens = len(tokenizer.encode(text))
        if total_tokens + tokens > max_tokens:
            break
        context_parts.append(text)
        total_tokens += tokens

    return "\n\n".join(context_parts)
