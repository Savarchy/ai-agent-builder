from pydantic import BaseModel, HttpUrl
from typing import Optional, List

class IngestTextIn(BaseModel):
    title: Optional[str] = None
    text: str
    url: Optional[str] = None

class Doc(BaseModel):
    title: str
    text: str
    url: str | None = None

class IngestURLIn(BaseModel):
    # accept raw string; we’ll normalize in the route
    url: str
    title: Optional[str] = None

class AskIn(BaseModel):
    question: str
    k: int | None = 6
    bot_id: Optional[str] = None   # <-- add this

class Citation(BaseModel):
    title: str
    url: str
    snippet: str
    score: float

class AskOut(BaseModel):
    answer: str
    citations: List[Citation] = []
