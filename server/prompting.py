# prompting.py

from typing import List, Tuple

def build_prompt(question: str, contexts: List[Tuple[str, str | None, float]]):
    """
    contexts: list of (text, url, score). We only pass the text to the LLM.
    Returns (system, user) messages for chat.
    """
    # Join top snippets (text only)
    snippets = []
    for text, _url, _score in contexts[:8]:
        if text:
            snippets.append(text.strip())
    joined = "\n\n---\n\n".join(snippets) if snippets else ""

    system = (
        "You are a helpful, honest assistant for a website.\n"
        "Use the provided CONTEXT when it is relevant.\n"
        "If the answer is not in the context, say you don't know.\n"
        "CRITICAL STYLE RULES:\n"
        "- Do NOT mention 'sources', 'documents', 'snippets', or 'context'.\n"
        "- Do NOT include citations, numbers in brackets, or footnotes (e.g., [1], [^1], (1)).\n"
        "- Do NOT invent links.\n"
        "- Answer directly and concisely in plain sentences.\n"
    )

    user = (
        f"CONTEXT:\n{joined}\n\n"
        f"QUESTION:\n{question}\n\n"
        "Answer plainly, with no citations or footnotes:"
    )

    return system, user
