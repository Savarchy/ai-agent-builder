# server/models.py
from sqlalchemy import Column, Integer, Text, DateTime, ForeignKey, func, text as sa_text
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from sqlalchemy.types import Float
from .db import Base

class Document(Base):
    __tablename__ = "documents"

    # DB: id uuid primary key default gen_random_uuid()
    id = Column(UUID(as_uuid=True), primary_key=True, server_default=sa_text("gen_random_uuid()"))
    title = Column(Text)
    url = Column(Text)
    # DB column is named 'text' (not 'content')
    text = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class Chunk(Base):
    __tablename__ = "chunks"

    id = Column(UUID(as_uuid=True), primary_key=True, server_default=sa_text("gen_random_uuid()"))
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"))
    ord = Column(Integer)
    text = Column(Text)
    # If you're storing arrays of floats; keep this if you aren't binding pgvector from SQLAlchemy.
    embedding = Column(ARRAY(Float))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
