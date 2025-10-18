# prompting.py
from typing import List, Tuple

def build_prompt(question: str, contexts: List[Tuple[str, str | None, float]]):
    """
    contexts: list of (text, url, score). We only pass the TEXT to the LLM.
    Returns (system, user) messages for chat.

    IMPORTANT: we render links ourselves in the UI. The model must NOT add
    links, citations, "Sources", footnotes, or bracketed numbers.
    """
    snippets: list[str] = []
    for text, _url, _score in contexts[:8]:
        if text:
            snippets.append(text.strip())
    joined = "\n\n---\n\n".join(snippets) if snippets else ""

    system = (
        "You are a careful, honest assistant.\n"
        "Use the CONTEXT when answering. If the answer is not in the context, say you don't know.\n"
        "STYLE RULES (MANDATORY):\n"
        "- Do NOT include citations, bracketed numbers, footnotes, or the word 'Sources'.\n"
        "- Do NOT include raw URLs or hyperlinks.\n"
        "- Answer plainly and concisely in complete sentences only.\n"
        "- After your final sentence, do not add anything else.\n"
    )

    user = (
        f"CONTEXT:\n{joined}\n\n"
        f"QUESTION:\n{question}\n\n"
        "Answer plainly with no links, no citations, and no footnotes:"
    )
    return system, user
