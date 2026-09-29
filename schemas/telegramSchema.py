from pydantic import BaseModel
from datetime import datetime


class TelegramIngestRequest(BaseModel):
    file_id: str
    filename: str
    description: str | None = None


class TelegramIngestResponse(BaseModel):
    filename: str
    table_name: str
    chunk_count: int
    is_update: bool
    updated_at: datetime


class TelegramDocumentItem(BaseModel):
    id: int
    filename: str
    table_name: str
    description: str | None
    chunk_count: int
    created_at: datetime
    updated_at: datetime


class TelegramBotDocument(BaseModel):
    file_id: str
    filename: str
    mime_type: str | None
    file_size: int | None
    date: int | None
