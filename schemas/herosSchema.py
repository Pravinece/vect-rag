from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class IngestRequest(BaseModel):
    filename: str
    description: Optional[str] = None


class MetadataRequest(BaseModel):
    filename: str


class IngestResponse(BaseModel):
    filename: str
    table_name: str
    chunk_count: int
    is_update: bool
    updated_at: datetime


class SourceRegistry(BaseModel):
    id: int
    filename: str
    table_name: str
    description: Optional[str]
    chunk_count: int
    created_at: datetime
    updated_at: datetime

class HeroSearchRequest(BaseModel):
    query: str
    top_k: int = 3


class SearchResult(BaseModel):
    filename: str
    chunk_id: int
    content: str
    score: float


class SearchResponse(BaseModel):
    query: str
    matched_source: str
    results: list[SearchResult]


class HeroChatRequest(BaseModel):
    question: str
    top_k: int = 3


class HeroChatResponse(BaseModel):
    question: str
    answer: str
    matched_source: str
    sources: list[SearchResult]