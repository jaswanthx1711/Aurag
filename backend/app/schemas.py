from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class RedactionFinding(BaseModel):
    start: int
    end: int
    entity_type: str
    text: str
    replacement: str
    confidence: float = 0.0
    method: str = "regex"


class ChunkCitation(BaseModel):
    chunk_id: str
    meeting_id: str
    meeting_title: str
    chunk_index: int
    score: float
    snippet: str
    keywords: list[str] = Field(default_factory=list)
    url: str | None = None               # present for web sources only
    source_type: str = "transcript"       # "transcript" | "web"


class QuestionRequest(BaseModel):
    question: str
    force_web: bool = False
    edit_index: int | None = None


class QuestionAnswer(BaseModel):
    question: str
    question_normalized: str
    answer: str
    confidence: float
    routing: str
    sources: list[ChunkCitation] = Field(default_factory=list)
    highlights: list[str] = Field(default_factory=list)
    created_at: datetime


class MeetingListItem(BaseModel):
    id: str
    title: str
    created_at: datetime
    audio_name: str | None = None
    processing_status: str
    quality_flag: str
    wer_estimate: float
    snr_estimate: float | None = None
    pii_count: int
    chunk_count: int
    question_count: int


class LectureNote(BaseModel):
    segment: int
    heading: str
    summary: str
    keywords: list[str] = Field(default_factory=list)
    text: str


class MeetingDetail(MeetingListItem):
    transcript: str
    sanitized_transcript: str
    speaker_transcript: str = ""
    summary: str
    action_items: list[str]
    topics: list[str]
    timeline: list[dict[str, Any]]
    redactions: list[RedactionFinding]
    questions: list[QuestionAnswer]
    model_trace: dict[str, Any]
    lecture_notes: list[LectureNote] = Field(default_factory=list)


class UploadResponse(BaseModel):
    meeting: MeetingDetail
    message: str


class DashboardResponse(BaseModel):
    total_meetings: int
    total_questions: int
    total_redactions: int
    average_wer: float
    average_snr: float | None = None
    latest_meetings: list[MeetingListItem] = Field(default_factory=list)
    topic_cloud: list[dict[str, Any]] = Field(default_factory=list)


class SearchHit(BaseModel):
    meeting_id: str
    meeting_title: str
    chunk_id: str
    chunk_index: int
    snippet: str
    score: float
    keywords: list[str] = Field(default_factory=list)


class SearchResponse(BaseModel):
    query: str
    corrected_query: str
    hits: list[SearchHit] = Field(default_factory=list)


# ── Auth ──────────────────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    email: str
    name: str
    password: str


class LoginRequest(BaseModel):
    email: str
    password: str


class UserInfo(BaseModel):
    id: str
    email: str
    name: str


class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserInfo


class ExportRequest(BaseModel):
    title: str
    text: str
    format: str
