from typing import Literal, Optional

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=8000)
    history: list[Message] = []
    use_docs: bool = False


class Source(BaseModel):
    n: int
    source: str
    page: Optional[int] = None
    snippet: str


class ChatResponse(BaseModel):
    reply: str
    sources: list[Source] = []


class DocumentInfo(BaseModel):
    id: str
    name: str
    chunks: int
