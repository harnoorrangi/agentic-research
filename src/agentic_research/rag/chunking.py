import re
from typing import List

from loguru import logger


def chunk_text(text: str, max_chars: int = 1200) -> List[str]:
    """Chunk text into smaller pieces based on sentence boundaries and a maximum character limit."""
    text = re.sub(r"\s+", " ", text).strip()
    sents = re.split(r"(?<=[\.\?\!])\s+", text)
    chunks, buf, cur = [], [], 0
    for s in sents:
        if cur + len(s) > max_chars and buf:
            chunks.append(" ".join(buf))
            buf, cur = [], 0
        buf.append(s)
        cur += len(s)
    if buf:
        chunks.append(" ".join(buf))
    logger.debug("chunk_text: produced {n} chunks for input length {l}", n=len(chunks), l=len(text))
    return chunks[:25]
