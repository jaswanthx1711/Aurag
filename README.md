# AuRAG: Audio Retrieval-Augmented Generation
### A Noise-Aware, Privacy-Preserving Meeting Intelligence System
**EPITA Graduate School of Computer Science — Spring 2026**

---

## 1. Project Overview

AuRAG is an advanced meeting analytics and voice-driven Retrieval-Augmented Generation (RAG) platform. It secures conversational data by applying inline PII redaction and corrects automatic speech recognition (ASR) anomalies before indexing files for semantic Q&A search. 

This repository was developed during the EPITA Action Learning MVP phase to transition empirical research findings into a production-ready application.

### Key Research Foundations Integrated:
1. **20% WER Quality Gate (Task 1 - Lakshithsaran):** Warns users of non-linear search quality degradation when Word Error Rate crosses the 20% threshold.
2. **Pre-Retrieval Spelling Correction (Task 2 - Jaswanth):** Implements SymSpell to resolve phonetic substitutions and punctuation loss, addressing the decoupled failure modes between retrieval and generation.
3. **Hybrid PII Redaction (Task 3 - Nishanth):** Merges spaCy Named Entity Recognition with Microsoft Presidio rule-based patterns to prevent PII leaks in ASR-corrupted transcripts.

---

## 2. Directory Layout & Tech Stack

```
Aurag/
├── backend/
│   ├── app/
│   │   ├── main.py          # FastAPI routes, routers, and CORS configurations.
│   │   ├── service.py       # Core logic: VAD, SNR, Whisper, PII, FAISS + BM25, and RAG.
│   │   ├── auth.py          # User authentication and JWT configuration.
│   │   ├── config.py        # Environment variables and configurations.
│   │   ├── schemas.py       # Pydantic schema serializers.
│   │   └── seed.py          # Mock dataset seeds.
│   ├── requirements.txt     # Python requirements.
│   └── data/                # SQLite DB and uploaded audio files.
└── nextjs-frontend/
    ├── components/
    │   └── App.tsx          # Main React workspace component & chat logs.
    ├── app/
    │   ├── globals.css      # Core Tailwind CSS & custom theme configurations.
    │   └── page.tsx         # Page entrypoint.
```

### Stack Details:
* **Frontend**: Next.js, React 18, Tailwind CSS, Server-Sent Events (SSE).
* **Backend**: FastAPI, SQLite / PostgreSQL.
* **ML Engines**: `faster-whisper`, `sentence-transformers` (`all-MiniLM-L6-v2`), `faiss-cpu`, `rank-bm25`, `presidio-analyzer`, `spaCy`, `llama-cpp-python` (Mistral-7B GGUF).

---

## 3. Configuration & Environment Variables

Create a `.env` file inside the `backend/` directory to customize configurations:

| Variable Name | Default Value | Purpose |
| :--- | :--- | :--- |
| `AURAG_DATABASE_URL` | `sqlite:///./data/aurag.db` | Target database connection URI (SQLite or PostgreSQL). |
| `AURAG_USE_API_MODELS` | `false` | Set to `true` to use cloud APIs instead of local models. |
| `AURAG_WHISPER_MODEL` | `small` | Core Whisper model size for ASR (`tiny`, `small`, `medium`). |
| `AURAG_LLM_MODEL_PATH` | *None* | Absolute path to the local Mistral-7B GGUF model file. |
| `GROQ_API_KEY` | *None* | API Key to run cloud models via Groq. |
| `OPENAI_API_KEY` | *None* | API Key to run cloud models via OpenAI. |
| `PORT` | `8000` | The backend API server listening port. |

---

## 4. Setup & Running Instructions

### A. Run Backend Services
1. Navigate to the backend directory:
   ```bash
   cd backend
   ```
2. Create and activate a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the development API server:
   ```bash
   python -m uvicorn app.main:app --reload --port 8000
   ```

The backend documentation will be accessible locally at [http://localhost:8000/docs](http://localhost:8000/docs).

### B. Run Frontend UI Workspace
1. Navigate to the frontend directory:
   ```bash
   cd nextjs-frontend
   ```
2. Install npm packages:
   ```bash
   npm install
   ```
3. Start the Next.js development server:
   ```bash
   npm run dev -- -p 3000
   ```

Open [http://localhost:3000](http://localhost:3000) in your web browser to access the workspace.

---

## 5. Deployed Application Features

### Core Features (CF)
* **CF1: Audio Upload & ASR** (ASR transcription, Voice Activity Detection, and estimated quality checks).
* **CF2: PII Redaction** (Scrubbing names, emails, and phones prior to serialization).
* **CF3: SymSpell Indexing** (ASR spelling corrections applied before search indexing).
* **CF4: Hybrid RAG Q&A** (Cosine vectors matched with BM25 keyword ranks).
* **CF5: Meeting Dashboard** (Global stats, quality scores, and extracted topic tag clouds).
* **CF6: Agentic Answer Engine** (Automatic failover to web search when local context is missing).
* **CF7: Action Items Checklist** (Task items, assignees, and deadlines checklists).
* **CF8: Topic Lecture Notes** (Segmented summaries with clickable timestamp blocks).

### Additional Features (AF)
* **AF1: WER Quality Indicators** (Badges indicating audio noise reliability status).
* **AF2: PII Redaction Audit PDF** (Generates downloadable PDF records of redacted text entities).
* **AF3: In-Browser Recording** (Microphone speech capture inside client browser workspace).
* **AF4: Q&A session export** (Download chat history logs to JSON / CSV).
* **Dropdown Theme Customization**: 9 developer layouts (*Dark+, Tokyo Night, Monokai, Solarized Dark/Light, Synthwave '84, Quiet Light, Light Modern, Tokyo Night Light*) selectable from the settings panel.

---

## 6. Documentation Index

For detailed explanation guides, refer to:
* **[PROJECT_DEFENCE.md](file:///Users/jaswanththathireddy/Desktop/Aurag/PROJECT_DEFENCE.md)**: A slide-by-slide script containing visuals, key points, and presenter scripts for your slide presentations.
* **[PROJECT_EXPLANATION.md](file:///Users/jaswanththathireddy/Desktop/Aurag/PROJECT_EXPLANATION.md)**: A comprehensive engineering reference guide detailing methods, mathematical equations (VAD, SNR, WER), database tables, and code snippet flows.
* **[deployment_guide.md](file:///Users/jaswanththathireddy/Desktop/Aurag/deployment_guide.md)**: Detailed hosting instructions (Render, Vercel, Supabase, ngrok).
