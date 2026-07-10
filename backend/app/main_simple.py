from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timezone

app = FastAPI(title="Aurag", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Demo data
DEMO_MEETINGS = [
    {
        "id": "demo-1",
        "title": "Project sync - retrieval pipeline",
        "audio_name": None,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "transcript": "Speaker A: We should ship the upload flow first...",
        "sanitized_transcript": "Speaker A: We should ship the upload flow first...",
        "summary": "The team agreed to prioritize upload",
        "action_items": ["Wire upload flow"],
        "topics": ["upload", "quality gate"],
        "timeline": [{"label": "Upload", "detail": "Audio enters the ASR pipeline."}],
        "redactions": [],
        "questions": [],
        "chunks": [],
        "model_trace": {},
        "processing_status": "ready",
        "quality_flag": "green",
        "wer_estimate": 0.08,
        "snr_estimate": 28.0,
        "pii_count": 0,
        "chunk_count": 0,
        "question_count": 0,
    },
    {
        "id": "demo-2",
        "title": "Research standup - mitigation strategies",
        "audio_name": None,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "transcript": "Speaker A: SymSpell improves noisy queries...",
        "sanitized_transcript": "Speaker A: SymSpell improves noisy queries...",
        "summary": "The team compared SymSpell strategies",
        "action_items": ["Compare strategies"],
        "topics": ["SymSpell", "BM25"],
        "timeline": [{"label": "Noise mitigation", "detail": "SymSpell and BM25 discussed."}],
        "redactions": [],
        "questions": [],
        "chunks": [],
        "model_trace": {},
        "processing_status": "ready",
        "quality_flag": "green",
        "wer_estimate": 0.1,
        "snr_estimate": 26.0,
        "pii_count": 0,
        "chunk_count": 0,
        "question_count": 0,
    },
]


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "Aurag"}


@app.get("/api/meetings")
def list_meetings():
    return [
        {
            "id": m["id"],
            "title": m["title"],
            "created_at": m["created_at"],
            "audio_name": m["audio_name"],
            "processing_status": m["processing_status"],
            "quality_flag": m["quality_flag"],
            "wer_estimate": m["wer_estimate"],
            "snr_estimate": m["snr_estimate"],
            "pii_count": m["pii_count"],
            "chunk_count": m["chunk_count"],
            "question_count": m["question_count"],
        }
        for m in DEMO_MEETINGS
    ]


@app.get("/api/meetings/{meeting_id}")
def get_meeting(meeting_id: str):
    for m in DEMO_MEETINGS:
        if m["id"] == meeting_id:
            return m
    return {"error": "Meeting not found"}


@app.get("/api/dashboard")
def dashboard():
    return {
        "total_meetings": len(DEMO_MEETINGS),
        "total_questions": 0,
        "total_redactions": 0,
        "average_wer": 0.09,
        "average_snr": 27.0,
        "latest_meetings": [
            {
                "id": m["id"],
                "title": m["title"],
                "created_at": m["created_at"],
                "audio_name": m["audio_name"],
                "processing_status": m["processing_status"],
                "quality_flag": m["quality_flag"],
                "wer_estimate": m["wer_estimate"],
                "snr_estimate": m["snr_estimate"],
                "pii_count": m["pii_count"],
                "chunk_count": m["chunk_count"],
                "question_count": m["question_count"],
            }
            for m in DEMO_MEETINGS[:2]
        ],
        "topic_cloud": [
            {"text": "upload", "value": 3},
            {"text": "quality", "value": 2},
            {"text": "retrieval", "value": 2},
        ],
    }
