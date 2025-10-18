# server/models.py
from .db import Base  # <-- make sure this import is present and at the top
import uuid
from datetime import datetime

from sqlalchemy import Column, Integer, Text, String, ForeignKey, DateTime
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector

from .db import Base  # <-- make sure this import is present and at the top

# If you use text-embedding-3-small, the vector size is 1536.
EMBED_DIM = 1536

class Document(Base):
    __tablename__ = "documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String, nullable=False)
    url = Column(String, nullable=True)
    text = Column(Text, nullable=False) 
    created_at = Column(DateTime, default=datetime.utcnow)

    chunks = relationship(
        "Chunk",
        cascade="all, delete-orphan",
        back_populates="document",
    )

class Chunk(Base):
    __tablename__ = "chunks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
    )
    ord = Column(Integer, nullable=False)
    text = Column(Text, nullable=False)

    # vector column for embeddings
    embedding = Column(Vector(EMBED_DIM))

    created_at = Column(DateTime, default=datetime.utcnow)

    document = relationship("Document", back_populates="chunks")
