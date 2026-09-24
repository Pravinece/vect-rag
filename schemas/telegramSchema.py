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
