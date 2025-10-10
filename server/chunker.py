import tiktoken
def split_text(text: str, chunk_size: int = 800, chunk_overlap: int = 120) -> list[str]:
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunks.append(enc.decode(chunk_tokens))
        start += max(1, chunk_size - chunk_overlap)
    return chunks
