# AuRAG Technology & Action Mapping Guide

This document maps every primary tool, framework, and library in the **AuRAG** codebase to its specific action and purpose in the system.

---

## 🗺️ High-Level Summary Table

| Library / Tool / Technology | System Layer | Action / Purpose in AuRAG |
| :--- | :--- | :--- |
| **FastAPI** | Backend | Serves REST API routes and handles Server-Sent Events (SSE) for streaming Q&A. |
| **Next.js & React** | Frontend | Powers the interactive single-page UI, meeting player, and real-time chat interface. |
| **Tailwind CSS** | Frontend | Manages responsive layouts and handles the 9 custom dark/light color themes. |
| **faster-whisper** | Backend | Core speech-to-text (ASR) transcription engine running on CPU. |
| **mlx-whisper** | Backend | Alternative speech-to-text engine optimized for Apple Silicon GPUs. |
| **Soundfile** | Backend | Reads and decodes raw binary WAV files into numerical arrays. |
| **NumPy** | Backend | Performs mathematical calculations on audio samples (RMS energy, downmixing, SNR calculations). |
| **SymSpell (`symspellpy`)** | Backend | Automatically corrects spelling/ASR typos in transcripts and incoming search queries. |
| **Microsoft Presidio** | Backend | Scans transcripts to detect sensitive Personally Identifiable Information (PII) like SSNs and wallets. |
| **spaCy (`en_core_web_sm`)** | Backend | Used by Presidio as the NLP engine to analyze text context and syntax. |
| **Sentence-Transformers** | Backend | Generates 384-dimensional dense vector embeddings using `all-MiniLM-L6-v2`. |
| **FAISS** | Backend | Performs fast vector similarity searches (dense retrieval) over transcript chunks. |
| **Rank-BM25** | Backend | Performs lexical keyword matching (sparse retrieval) over transcripts. |
| **Llama-cpp-python** | Backend | Loads and runs the local 4.1GB Mistral-7B GGUF model with Metal GPU acceleration. |
| **SQLite (`sqlite3`)** | Backend | Serializes and stores meetings, chunks, PII redactions, and Q&A history. |
| **Cloudflare Tunnel** | Deployment | Exposes the local dev ports to a secure public URL for mobile testing. |
| **Argon2** | Backend | Secures user passwords using advanced hashing functions. |
| **PyJWT** | Backend | Encodes, decodes, and validates JWT authentication session tokens. |

---

## 🔍 Detailed Breakdown of Actions

### 1. Audio Processing & Quality Assessment
* **soundfile**: Reads incoming uploaded WAV audio files.
* **numpy**:
  * Downmixes multi-channel stereo files to mono by taking the mean of sample arrays.
  * Splits samples into 20ms frames and computes Root Mean Square (RMS) energy to detect speech vs. silence.
  * Measures voice signal amplitude vs. noise floor to calculate the Signal-to-Noise Ratio (SNR) in decibels (dB).

### 2. Automatic Speech Recognition (ASR)
* **faster-whisper**: Uses CTranslate2 to run OpenAI's Whisper model efficiently on CPU, transcribing audio streams into time-stamped text segments.
* **mlx-whisper**: Integrates with Apple's MLX framework to utilize Apple Silicon GPU cores for accelerated transcription when available.

### 3. Spelling & Transcription Corrections
* **SymSpell (`symspellpy`)**: Corrects spelling anomalies resulting from accents, background noise, or model errors. It loops over words with an edit distance of up to 2, correcting the transcript before storage and aligning user queries to transcript contents during searches.

### 4. Personally Identifiable Information (PII) Redaction
* **Microsoft Presidio & spaCy**: Analyzes context to locate sensitive data (such as emails, phones, SSNs, credit cards, passport numbers, and crypto wallets). Spans are masked with tokens (e.g. `[EMAIL_ADDRESS]`) to ensure no PII enters SQLite.

### 5. Semantic Indexing & Hybrid Search (RAG)
* **Sentence-Transformers (`all-MiniLM-L6-v2`)**: Generates mathematical vector representations of semantic context.
* **FAISS**: Stores dense vectors and performs cosine-similarity search to retrieve chunks based on meaning.
* **Rank-BM25**: Indexes text keywords to perform keyword-frequency matching.
* **Hybrid Search Ranker**: Merges results from FAISS and Rank-BM25 using Reciprocal Rank Fusion (RRF) to select the top 5 most relevant context chunks.

### 6. Answer Generation
* **Llama-cpp-python**: Loads the local GGUF Mistral model. When a question is asked, it runs Metal GPU-accelerated text generation to construct a natural summary response from the RAG chunks.

### 7. Networking & Accessibility
* **Cloudflare Tunnel (`cloudflared`)**: Creates a secure bridge between `http://localhost:3000` and `https://*.trycloudflare.com`, routing external mobile browser requests to your local computer.
