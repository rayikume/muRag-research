from typing import Literal

from pydantic import BaseModel, Field


class ToolUsage(BaseModel):
    name: str
    input: str
    output: str


class RetrievedChunk(BaseModel):
    text: str
    source: str
    score: float


class AgentResponse(BaseModel):
    finalRes: str
    agentUsed: Literal["General", "RAG"]
    toolsUsed: list[ToolUsage] = Field(default_factory=list)
    retrievedChunks: list[RetrievedChunk] = Field(default_factory=list)


class QueryRequest(BaseModel):
    query: str
