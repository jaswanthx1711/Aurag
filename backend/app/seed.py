from __future__ import annotations

from datetime import datetime, timezone


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


DEMO_MEETINGS = [
    {
        "id": "demo-alpha",
        "title": "Project sync - retrieval pipeline and deck review",
        "audio_name": None,
        "created_at": _now(),
        "transcript": (
            "Speaker A: We should ship the upload flow first and warn users when audio quality is poor. "
            "Speaker B: Agreed, the quality gate should flag noisy transcripts before indexing. "
            "Speaker A: The RAG answers need sources and the dashboard should show WER, PII count, and question history. "
            "Speaker B: We also need a redaction report for emails like nishanth@example.com and phone numbers such as +33 6 12 34 56 78. "
            "Speaker A: The next step is to wire the meeting view to the hybrid retriever and keep the UI clean."
        ),
        "summary": "The team agreed to prioritize upload, quality gating, source-backed answers, and the dashboard metrics.",
        "action_items": [
            "Wire upload flow to the ASR pipeline.",
            "Surface WER, PII count, and answer history in the dashboard.",
            "Connect Q&A to the hybrid retriever and citations."
        ],
        "topics": ["upload", "quality gate", "redaction", "hybrid retrieval", "dashboard"],
        "timeline": [
            {"label": "Upload", "detail": "Audio enters the ASR pipeline."},
            {"label": "Redact", "detail": "PII spans are masked before storage."},
            {"label": "Index", "detail": "Chunks are embedded and indexed."},
            {"label": "Answer", "detail": "Hybrid RAG returns grounded responses."},
        ],
    },
    {
        "id": "demo-beta",
        "title": "Research standup - mitigation strategies",
        "audio_name": None,
        "created_at": _now(),
        "transcript": (
            "Speaker A: SymSpell improves noisy queries, and BM25 helps when the transcript is sparse. "
            "Speaker B: Semantic chunking recovered recall better than fixed windows in the notebook run. "
            "Speaker A: We should keep the confidence badge on low-support answers and store the follow-up history. "
            "Speaker B: Action item: compare the clean baseline with WER thirty five percent on the next slide deck."
        ),
        "summary": "The team compared SymSpell, hybrid retrieval, and semantic chunking as mitigation strategies.",
        "action_items": [
            "Compare mitigation strategies in the deck.",
            "Keep confidence badges visible on answers.",
            "Persist follow-up questions for the meeting view."
        ],
        "topics": ["SymSpell", "BM25", "semantic chunking", "confidence", "follow-up history"],
        "timeline": [
            {"label": "Noise mitigation", "detail": "SymSpell and BM25 are discussed."},
            {"label": "Chunking", "detail": "Semantic windows improve retrieval recall."},
            {"label": "Trust", "detail": "Confidence badges and history are emphasized."},
        ],
    },
    {
        "id": "demo-gamma",
        "title": "Launch planning - exports and sharing",
        "audio_name": None,
        "created_at": _now(),
        "transcript": (
            "Speaker A: We need PDF export for the answer and a clean shareable meeting link. "
            "Speaker B: Later we can add DOCX and image export, plus a Slack digest for the team. "
            "Speaker A: The backlog should also cover saved questions and multi-meeting search. "
            "Speaker B: For now, the main feature is a stable meeting dashboard with history."
        ),
        "summary": "The launch plan focused on exports, sharing, backlog features, and dashboard stability.",
        "action_items": [
            "Add PDF export for answers.",
            "Support shareable meeting links and team digests.",
            "Track backlog items for search and favorites."
        ],
        "topics": ["exports", "sharing", "Slack digest", "search", "saved questions"],
        "timeline": [
            {"label": "Export", "detail": "PDF and future formats are scoped."},
            {"label": "Share", "detail": "Meeting links and digests are discussed."},
            {"label": "Backlog", "detail": "Search and favorites are parked for later."},
        ],
    },
]
