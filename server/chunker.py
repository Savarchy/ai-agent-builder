# chunker.py
import os
import re
from typing import List, Tuple

CHUNK_MAX = int(os.getenv("CHUNK_MAX_CHARS", "3000"))        # default ~3k chars
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP_CHARS", "300")) # default 10% overlap

_SENT_SPLIT = re.compile(r'(?<=[\.\!\?])\s+')
_PARA_SPLIT = re.compile(r'\n{2,}')

def _split_sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]

def split_text(text: str) -> Tuple[List[str], None]:
    """
    Returns (chunks, None).
    Greedy paragraph → sentence packing with overlap.
    """
    if not text:
        return ([], None)

    paras = [p.strip() for p in _PARA_SPLIT.split(text) if p.strip()]
    chunks: List[str] = []
    buf: List[str] = []
    size = 0

    def flush():
        nonlocal buf, size
        if not buf:
            return
        chunk = " ".join(buf).strip()
        if chunk:
            chunks.append(chunk)
        # create overlap window
        overlap_chars = CHUNK_OVERLAP
        if overlap_chars > 0 and chunk:
            # keep tail of current chunk for overlap
            tail = chunk[-overlap_chars:]
            buf = [tail]
            size = len(tail)
        else:
            buf = []
            size = 0

    for p in paras:
        sents = _split_sentences(p) or [p]
        for s in sents:
            s_len = len(s) + 1
            if size + s_len > CHUNK_MAX and size > 0:
                flush()
            buf.append(s)
            size += s_len
            if size >= CHUNK_MAX:
                flush()

        # soft break between paragraphs
        if size > 0 and (size + 1 <= CHUNK_MAX):
            buf.append("\n")
            size += 1

    if buf:
        flush()

    return (chunks or [text], None)
