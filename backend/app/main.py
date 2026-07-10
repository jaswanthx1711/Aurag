from __future__ import annotations

import io
import threading
import uuid
from collections import Counter
from datetime import datetime, timezone
from typing import Annotated

# In-memory cache for one-time export download tokens
_export_cache: dict[str, dict] = {}

from fastapi import Depends, FastAPI, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse

from .auth import create_token, get_current_user, hash_password, verify_password
from .config import settings
from .schemas import (
    AuthResponse, DashboardResponse, LoginRequest, MeetingDetail, MeetingListItem,
    QuestionAnswer, QuestionRequest, RegisterRequest, SearchResponse, UploadResponse, UserInfo,
    ExportRequest,
)
from .service import SERVICE

app = FastAPI(title=settings.app_name, version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── helpers ────────────────────────────────────────────────────────────────────

def _uid(user: dict) -> str:
    return user["sub"]

# ── public ─────────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": settings.app_name}


@app.post("/api/auth/register", response_model=AuthResponse)
def register(payload: RegisterRequest) -> AuthResponse:
    if not payload.email or "@" not in payload.email:
        raise HTTPException(status_code=400, detail="Invalid email address")
    if len(payload.password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters")
    if not payload.name.strip():
        raise HTTPException(status_code=400, detail="Name is required")
    try:
        user = SERVICE.create_user(payload.email, payload.name, hash_password(payload.password))
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e)) from None
    token = create_token(user["id"], user["email"], user["name"])
    return AuthResponse(access_token=token, user=UserInfo(**user))


@app.post("/api/auth/login", response_model=AuthResponse)
def login(payload: LoginRequest) -> AuthResponse:
    user = SERVICE.get_user_by_email(payload.email)
    if user is None or not verify_password(payload.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    token = create_token(user["id"], user["email"], user["name"])
    return AuthResponse(
        access_token=token,
        user=UserInfo(id=user["id"], email=user["email"], name=user["name"]),
    )


@app.get("/api/auth/me", response_model=UserInfo)
def me(current_user: dict = Depends(get_current_user)) -> UserInfo:
    user = SERVICE.get_user_by_id(_uid(current_user))
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return UserInfo(id=user["id"], email=user["email"], name=user["name"])

# ── protected meeting routes ───────────────────────────────────────────────────

@app.get("/api/meetings")
def list_meetings(current_user: dict = Depends(get_current_user)) -> list[MeetingListItem]:
    items = []
    for m in SERVICE.list_meetings(user_id=_uid(current_user)):
        items.append(MeetingListItem(
            id=m["id"], title=m["title"], created_at=m["created_at"],
            audio_name=m.get("audio_name"),
            processing_status=m.get("processing_status", "ready"),
            quality_flag=m.get("quality_flag", "green"),
            wer_estimate=m.get("wer_estimate", 0.0),
            snr_estimate=m.get("snr_estimate"),
            pii_count=m.get("pii_count", 0),
            chunk_count=len(m.get("chunks", [])),
            question_count=len(m.get("questions", [])),
        ))
    return items


@app.get("/api/meetings/{meeting_id}", responses={404: {"description": "Meeting not found"}})
def get_meeting(meeting_id: str, current_user: dict = Depends(get_current_user)) -> MeetingDetail:
    meeting = SERVICE.get_meeting(meeting_id, user_id=_uid(current_user))
    if meeting is None:
        raise HTTPException(status_code=404, detail="Meeting not found")
    meeting["question_count"] = len(meeting.get("questions", []))
    meeting["chunk_count"] = len(meeting.get("chunks", []))
    return MeetingDetail(**meeting)


@app.post("/api/meetings/{meeting_id}/questions", responses={404: {"description": "Meeting not found"}})
def ask_meeting_question(
    meeting_id: str, payload: QuestionRequest,
    current_user: dict = Depends(get_current_user),
) -> QuestionAnswer:
    meeting = SERVICE.get_meeting(meeting_id, user_id=_uid(current_user))
    if meeting is None:
        raise HTTPException(status_code=404, detail="Meeting not found")
    try:
        answer = SERVICE.answer_question(meeting_id, payload.question, force_web=payload.force_web)
    except KeyError:
        raise HTTPException(status_code=404, detail="Meeting not found") from None
    return QuestionAnswer(**answer)


@app.post("/api/meetings/{meeting_id}/questions/stream")
def ask_meeting_question_stream(
    meeting_id: str, payload: QuestionRequest,
    current_user: dict = Depends(get_current_user),
) -> StreamingResponse:
    import json as _json
    if SERVICE.get_meeting(meeting_id, user_id=_uid(current_user)) is None:
        raise HTTPException(status_code=404, detail="Meeting not found")
    try:
        ctx = SERVICE._route_question(meeting_id, payload.question, force_web=payload.force_web)
    except KeyError:
        raise HTTPException(status_code=404, detail="Meeting not found") from None

    meeting = ctx["meeting"]
    corrected_query = ctx["corrected_query"]

    def event_stream():
        yield f"data: {_json.dumps({'type': 'meta', 'routing': ctx['routing_label'], 'confidence': ctx['confidence'], 'sources': ctx['citations']})}\n\n"
        full_tokens: list[str] = []
        
        # Check if the user is requesting an export document format in the query
        is_export_request = False
        import re as _re
        export_patterns = [r"\b(pdf|docx?|csv|word|doc|export|download)\b"]
        if any(_re.search(pat, payload.question.lower()) for pat in export_patterns):
            is_export_request = True

        if is_export_request:
            fmt_match = _re.search(r"\b(pdf|docx?|csv|word|doc|txt|text)\b", payload.question.lower())
            fmt_str = fmt_match.group(1).upper() if fmt_match else "document"
            if fmt_str in ["DOC", "DOCX", "WORD"]:
                fmt_str = "Word document"
            elif fmt_str in ["TXT", "TEXT"]:
                fmt_str = "text file"
            confirmation = f"I've compiled the {fmt_str} for you! You can download the report below."
            for word in confirmation.split():
                tok = word + " "
                full_tokens.append(tok)
                yield f"data: {_json.dumps({'type': 'token', 'text': tok})}\n\n"
            yield f"data: {_json.dumps({'type': 'token', 'text': '\n'})}\n\n"
        elif ctx.get("is_greeting"):
            greeting_resp = "Hello! How can I help you with this meeting today?"
            for word in greeting_resp.split():
                tok = word + " "; full_tokens.append(tok)
                yield f"data: {_json.dumps({'type': 'token', 'text': tok})}\n\n"
        elif ctx["use_web"]:
            if ctx["web_hits"]:
                for tok in SERVICE._llm_stream(corrected_query, ctx["chunks"], meeting=meeting, web_hits=ctx["web_hits"]):
                    full_tokens.append(tok)
                    yield f"data: {_json.dumps({'type': 'token', 'text': tok})}\n\n"
            else:
                web_text = "Web search found no results for this query. Try rephrasing your question."
                for word in web_text.split():
                    tok = word + " "; full_tokens.append(tok)
                    yield f"data: {_json.dumps({'type': 'token', 'text': tok})}\n\n"
        elif not ctx["chunks"] and not ctx.get("support_ok", True):
            # Truly nothing to work with (no chunks at all) - this is the only case where we
            # short-circuit; any question with actual transcript content gets a real LLM attempt,
            # since the model itself is instructed to say when it can't find the answer.
            fallback = SERVICE._no_transcript_answer(force_web=False)
            for word in fallback.split():
                tok = word + " "; full_tokens.append(tok)
                yield f"data: {_json.dumps({'type': 'token', 'text': tok})}\n\n"
        else:
            for tok in SERVICE._llm_stream(corrected_query, ctx["chunks"], meeting=meeting):
                full_tokens.append(tok)
                yield f"data: {_json.dumps({'type': 'token', 'text': tok})}\n\n"
        SERVICE._save_question_answer(meeting, payload.question, corrected_query, "".join(full_tokens).strip(), ctx, edit_index=payload.edit_index)
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/api/upload")
async def upload_meeting(
    file: UploadFile,
    title: Annotated[str | None, Form()] = None,
    current_user: dict = Depends(get_current_user),
) -> UploadResponse:
    user_id = _uid(current_user)
    destination = settings.uploads_dir / file.filename
    with destination.open("wb") as handle:
        while chunk := await file.read(1024 * 1024):
            handle.write(chunk)

    stub = SERVICE.create_processing_stub(destination, title=title)
    stub["user_id"] = user_id
    SERVICE._write_meeting_to_db(stub)
    stub["question_count"] = 0
    stub["chunk_count"] = 0

    def _process():
        SERVICE.process_audio_async(stub["id"], destination, title=title, user_id=user_id)

    threading.Thread(target=_process, daemon=True).start()
    return UploadResponse(meeting=MeetingDetail(**stub), message="Upload received - transcription running in background.")


@app.post("/api/meetings/{meeting_id}/cancel")
def cancel_meeting_transcription(
    meeting_id: str, current_user: dict = Depends(get_current_user),
) -> dict[str, str]:
    if SERVICE.get_meeting(meeting_id, user_id=_uid(current_user)) is None:
        raise HTTPException(status_code=404, detail="Meeting not found")
    SERVICE.cancel_transcription(meeting_id)
    return {"status": "cancelled"}


@app.get("/api/dashboard")
def dashboard(current_user: dict = Depends(get_current_user)) -> DashboardResponse:
    return DashboardResponse(**SERVICE.dashboard(user_id=_uid(current_user)))


@app.get("/api/search")
def search(query: str, current_user: dict = Depends(get_current_user)) -> SearchResponse:
    return SearchResponse(**SERVICE.search(query, user_id=_uid(current_user)))


# ── report endpoints ────────────────────────────────────────────────────────────

@app.get("/api/meetings/{meeting_id}/redaction-report")
def redaction_report(meeting_id: str, current_user: dict = Depends(get_current_user)) -> Response:
    meeting = SERVICE.get_meeting(meeting_id, user_id=_uid(current_user))
    if meeting is None:
        raise HTTPException(status_code=404, detail="Meeting not found")
    try:
        from fpdf import FPDF  # type: ignore
    except ImportError:
        raise HTTPException(status_code=500, detail="fpdf2 not installed") from None

    redactions = meeting.get("redactions", [])
    title = meeting.get("title", meeting_id)
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 10, "AuRAG PII Redaction Report", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 7, f"Meeting: {title}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 7, f"Generated: {generated_at}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 7, f"Total PII findings: {len(redactions)}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    if not redactions:
        pdf.set_font("Helvetica", "I", 11)
        pdf.cell(0, 10, "No PII entities were detected in this meeting.", new_x="LMARGIN", new_y="NEXT")
    else:
        by_type = Counter(r.get("entity_type", "UNKNOWN") for r in redactions)
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 8, "Summary by Entity Type", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        for entity_type, count in sorted(by_type.items()):
            pdf.cell(0, 6, f"  {entity_type}: {count}", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(4)
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 8, "Detailed Findings", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "B", 9)
        col_w = [14, 36, 36, 26, 22, 26]
        for w, h in zip(col_w, ["#", "Entity Type", "Original Text", "Replacement", "Confidence", "Method"]):
            pdf.cell(w * 1.9, 7, h, border=1)
        pdf.ln()
        pdf.set_font("Helvetica", "", 8)
        for i, r in enumerate(redactions, 1):
            for w, cell in zip(col_w, [str(i), str(r.get("entity_type", ""))[:18], str(r.get("text", ""))[:20],
                                        str(r.get("replacement", ""))[:14], f"{float(r.get('confidence', 0)) * 100:.0f}%",
                                        str(r.get("method", ""))[:12]]):
                pdf.cell(w * 1.9, 6, cell, border=1)
            pdf.ln()

    content = bytes(pdf.output())
    safe_title = "".join(c if c.isalnum() or c in " _-" else "_" for c in title)[:40]
    return Response(content, media_type="application/pdf",
                    headers={"Content-Disposition": f'attachment; filename="redaction-report-{safe_title}.pdf"'})


@app.get("/api/meetings/{meeting_id}/summary-report")
def summary_report(meeting_id: str, current_user: dict = Depends(get_current_user)) -> Response:
    meeting = SERVICE.get_meeting(meeting_id, user_id=_uid(current_user))
    if meeting is None:
        raise HTTPException(status_code=404, detail="Meeting not found")
    try:
        from fpdf import FPDF  # type: ignore
        from fpdf.enums import XPos, YPos  # type: ignore
    except ImportError:
        raise HTTPException(status_code=500, detail="fpdf2 not installed") from None

    def safe(text: str) -> str:
        return text.encode("latin-1", errors="replace").decode("latin-1")

    NL = dict(new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    title = safe(meeting.get("title", meeting_id))
    summary = safe(meeting.get("summary", "") or "No summary available.")
    action_items = [safe(a) for a in (meeting.get("action_items") or [])]
    topics = [safe(t) for t in (meeting.get("topics") or [])]
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Helvetica", "B", 18)
    pdf.multi_cell(0, 10, title)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(120, 140, 160)
    pdf.cell(0, 7, f"AuRAG Meeting Report  |  {generated_at}", **NL)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(5)
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "Summary", **NL)
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(0, 7, summary, **NL)
    pdf.ln(4)
    if action_items:
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 8, "Action Items", **NL)
        pdf.set_font("Helvetica", "", 11)
        for item in action_items:
            pdf.multi_cell(0, 7, f"- {item}", **NL)
        pdf.ln(4)
    if topics:
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 8, "Key Topics", **NL)
        pdf.set_font("Helvetica", "", 11)
        pdf.multi_cell(0, 7, ", ".join(topics), **NL)

    content = bytes(pdf.output())
    safe_title = "".join(c if c.isalnum() or c in " _-" else "_" for c in title)[:40]
    return Response(content, media_type="application/pdf",
                    headers={"Content-Disposition": f'attachment; filename="summary-{safe_title}.pdf"'})


@app.get("/api/meetings/{meeting_id}/summary-docx")
def summary_docx(meeting_id: str, current_user: dict = Depends(get_current_user)) -> Response:
    meeting = SERVICE.get_meeting(meeting_id, user_id=_uid(current_user))
    if meeting is None:
        raise HTTPException(status_code=404, detail="Meeting not found")
    try:
        from docx import Document  # type: ignore
        from docx.shared import Pt, RGBColor  # type: ignore
    except ImportError:
        raise HTTPException(status_code=500, detail="python-docx not installed") from None

    title = meeting.get("title", meeting_id)
    summary = meeting.get("summary", "") or "No summary available."
    action_items = meeting.get("action_items") or []
    topics = meeting.get("topics") or []
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    doc = Document()
    heading = doc.add_heading(title, level=0)
    heading.runs[0].font.color.rgb = RGBColor(0x1A, 0x8A, 0x7A)
    meta = doc.add_paragraph(f"AuRAG Meeting Report  |  {generated_at}")
    meta.runs[0].font.size = Pt(9)
    meta.runs[0].font.color.rgb = RGBColor(0x7A, 0x9A, 0xAA)
    doc.add_heading("Summary", level=2)
    doc.add_paragraph(summary)
    if action_items:
        doc.add_heading("Action Items", level=2)
        for item in action_items:
            doc.add_paragraph(item, style="List Bullet")
    if topics:
        doc.add_heading("Key Topics", level=2)
        doc.add_paragraph(", ".join(topics))

    buf = io.BytesIO()
    doc.save(buf)
    safe_title = "".join(c if c.isalnum() or c in " _-" else "_" for c in title)[:40]
    return Response(buf.getvalue(),
                    media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    headers={"Content-Disposition": f'attachment; filename="summary-{safe_title}.docx"'})


@app.get("/api/meetings/{meeting_id}/summary-txt")
def summary_txt(meeting_id: str, current_user: dict = Depends(get_current_user)) -> Response:
    meeting = SERVICE.get_meeting(meeting_id, user_id=_uid(current_user))
    if meeting is None:
        raise HTTPException(status_code=404, detail="Meeting not found")

    buf = f"AuRAG Meeting Summary - {meeting.get('title', meeting_id)}\n"
    buf += f"Created at: {meeting.get('created_at')}\n"
    buf += "=" * 60 + "\n\n"
    
    buf += "# SUMMARY\n"
    buf += meeting.get("summary", "") + "\n\n"
    
    action_items = meeting.get("action_items") or []
    if action_items:
        buf += "# ACTION ITEMS\n"
        for item in action_items:
            buf += f"- {item}\n"
        buf += "\n"
        
    topics = meeting.get("topics") or []
    if topics:
        buf += "# TOPICS\n"
        buf += ", ".join(topics) + "\n\n"
        
    questions = meeting.get("questions") or []
    if questions:
        buf += "# Q&A HISTORY\n"
        for q in reversed(questions):
            buf += f"Q: {q.get('question')}\n"
            buf += f"A: {q.get('answer')}\n\n"

    safe_title = "".join(c if c.isalnum() or c in " _-" else "_" for c in meeting.get("title", meeting_id))[:40]
    return Response(buf, media_type="text/plain",
                    headers={"Content-Disposition": f'attachment; filename="summary-{safe_title}.txt"'})


@app.post("/api/export")
def export_text(payload: ExportRequest, current_user: dict = Depends(get_current_user)) -> Response:
    try:
        from fpdf import FPDF  # type: ignore
        from fpdf.enums import XPos, YPos  # type: ignore
    except ImportError:
        pass

    try:
        from docx import Document  # type: ignore
        from docx.shared import Pt, RGBColor  # type: ignore
    except ImportError:
        pass

    fmt = payload.format.lower()
    title = payload.title
    text = payload.text

    # Intercept short confirmations and replace with the full detailed meeting report summary
    if len(text) < 250 or "download the report" in text.lower() or "compiled" in text.lower():
        matched_m = None
        for m in SERVICE.meetings:
            m_title = m.get("title", "").lower().strip()
            t_title = title.lower().strip()
            if m_title == t_title or t_title in m_title or m_title in t_title:
                matched_m = m
                break
        if not matched_m and SERVICE.meetings:
            matched_m = SERVICE.meetings[0]
        if matched_m:
            text = SERVICE._answer_from_meeting_structure(matched_m)

    def safe(txt: str) -> str:
        txt = txt.replace("•", "\u00b7")
        return txt.encode("latin-1", errors="replace").decode("latin-1")

    safe_title = "".join(c if c.isalnum() or c in " _-" else "_" for c in title)[:40]

    if fmt == "pdf":
        NL = dict(new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.set_font("Helvetica", "B", 16)
        pdf.multi_cell(0, 10, safe(title))
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(120, 140, 160)
        pdf.cell(0, 7, f"AuRAG Generated Report  |  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}", **NL)
        pdf.set_text_color(0, 0, 0)
        pdf.ln(8)
        
        pdf.set_font("Helvetica", "", 11)
        for line in text.split("\n"):
            line = line.strip()
            if not line:
                pdf.ln(4)
                continue
            if line.startswith("#"):
                pdf.set_font("Helvetica", "B", 13)
                pdf.multi_cell(0, 8, safe(line.lstrip("#").strip()), **NL)
                pdf.set_font("Helvetica", "", 11)
            elif line.startswith("-") or line.startswith("*") or line.startswith("•"):
                pdf.multi_cell(0, 7, safe(f"  \u00b7  {line[1:].strip()}"), **NL)
            else:
                pdf.multi_cell(0, 7, safe(line), **NL)

        content = bytes(pdf.output())
        return Response(content, media_type="application/pdf",
                        headers={"Content-Disposition": f'attachment; filename="export-{safe_title}.pdf"'})

    elif fmt == "docx":
        doc = Document()
        heading = doc.add_heading(title, level=0)
        heading.runs[0].font.color.rgb = RGBColor(0x1A, 0x8A, 0x7A)
        meta = doc.add_paragraph(f"AuRAG Generated Report  |  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
        meta.runs[0].font.size = Pt(9)
        meta.runs[0].font.color.rgb = RGBColor(0x7A, 0x9A, 0xAA)

        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                h = doc.add_heading(line.lstrip("#").strip(), level=2)
                h.runs[0].font.color.rgb = RGBColor(0x1A, 0x8A, 0x7A)
            elif line.startswith("-") or line.startswith("*") or line.startswith("•"):
                doc.add_paragraph(line[1:].strip(), style="List Bullet")
            else:
                doc.add_paragraph(line)

        buf = io.BytesIO()
        doc.save(buf)
        return Response(buf.getvalue(),
                        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        headers={"Content-Disposition": f'attachment; filename="export-{safe_title}.docx"'})

    elif fmt == "csv":
        import csv
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["AuRAG Generated Export - " + title])
        writer.writerow([])
        for line in text.split("\n"):
            writer.writerow([line])
        return Response(buf.getvalue(), media_type="text/csv",
                        headers={"Content-Disposition": f'attachment; filename="export-{safe_title}.csv"'})

    elif fmt in ["txt", "text"]:
        buf = f"AuRAG Generated Export - {title}\n"
        buf += f"Generated at: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n"
        buf += "=" * 50 + "\n\n"
        buf += text
        return Response(buf, media_type="text/plain",
                        headers={"Content-Disposition": f'attachment; filename="export-{safe_title}.txt"'})

    else:
        raise HTTPException(status_code=400, detail=f"Unsupported format: {fmt}")


@app.post("/api/export-form")
def export_form(
    title: str = Form(...),
    text: str = Form(...),
    format: str = Form(...),
    current_user: dict = Depends(get_current_user),
) -> Response:
    """Accept form-encoded POST (from a native HTML form) and return a file download."""
    from .schemas import ExportRequest as _ER
    return export_text(_ER(title=title, text=text, format=format), current_user)


@app.post("/api/export-url")
def export_url(payload: ExportRequest, current_user: dict = Depends(get_current_user)) -> dict:
    """Store export payload in memory with a one-time token. Returns a URL the browser can navigate to directly."""
    token = str(uuid.uuid4())
    _export_cache[token] = {
        "title": payload.title,
        "text": payload.text,
        "format": payload.format,
    }
    return {"url": f"/api/download/{token}"}


@app.get("/api/download/{token}")
def download_export(token: str) -> Response:
    """Serve the cached export as a real file download (no auth needed — token is the secret)."""
    data = _export_cache.pop(token, None)
    if data is None:
        raise HTTPException(status_code=404, detail="Download link expired or not found.")

    try:
        from fpdf import FPDF  # type: ignore
        from fpdf.enums import XPos, YPos  # type: ignore
    except ImportError:
        pass
    try:
        from docx import Document  # type: ignore
        from docx.shared import Pt, RGBColor  # type: ignore
    except ImportError:
        pass

    fmt = data["format"].lower()
    title = data["title"]
    text = data["text"]

    # Intercept short confirmations and replace with the full detailed meeting report summary
    if len(text) < 250 or "download the report" in text.lower() or "compiled" in text.lower():
        matched_m = None
        for m in SERVICE.meetings:
            m_title = m.get("title", "").lower().strip()
            t_title = title.lower().strip()
            if m_title == t_title or t_title in m_title or m_title in t_title:
                matched_m = m
                break
        if not matched_m and SERVICE.meetings:
            matched_m = SERVICE.meetings[0]
        if matched_m:
            text = SERVICE._answer_from_meeting_structure(matched_m)

    def safe(txt: str) -> str:
        txt = txt.replace("•", "\u00b7")
        return txt.encode("latin-1", errors="replace").decode("latin-1")

    safe_title = "".join(c if c.isalnum() or c in " _-" else "_" for c in title)[:40]

    if fmt == "pdf":
        NL = dict(new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.set_font("Helvetica", "B", 16)
        pdf.multi_cell(0, 10, safe(title))
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(120, 140, 160)
        pdf.cell(0, 7, f"AuRAG Report  |  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}", **NL)
        pdf.set_text_color(0, 0, 0)
        pdf.ln(8)
        pdf.set_font("Helvetica", "", 11)
        for line in text.split("\n"):
            line = line.strip()
            if not line:
                pdf.ln(4)
                continue
            if line.startswith("#"):
                pdf.set_font("Helvetica", "B", 13)
                pdf.multi_cell(0, 8, safe(line.lstrip("#").strip()), **NL)
                pdf.set_font("Helvetica", "", 11)
            elif line.startswith("-") or line.startswith("*") or line.startswith("•"):
                pdf.multi_cell(0, 7, safe(f"  \u00b7  {line[1:].strip()}"), **NL)
            else:
                pdf.multi_cell(0, 7, safe(line), **NL)
        content = bytes(pdf.output())
        return Response(content, media_type="application/pdf",
                        headers={"Content-Disposition": f'attachment; filename="export-{safe_title}.pdf"'})

    elif fmt == "docx":
        doc = Document()
        heading = doc.add_heading(title, level=0)
        heading.runs[0].font.color.rgb = RGBColor(0x1A, 0x8A, 0x7A)
        meta = doc.add_paragraph(f"AuRAG Report  |  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
        meta.runs[0].font.size = Pt(9)
        meta.runs[0].font.color.rgb = RGBColor(0x7A, 0x9A, 0xAA)
        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                h = doc.add_heading(line.lstrip("#").strip(), level=2)
                h.runs[0].font.color.rgb = RGBColor(0x1A, 0x8A, 0x7A)
            elif line.startswith("-") or line.startswith("*") or line.startswith("•"):
                doc.add_paragraph(line[1:].strip(), style="List Bullet")
            else:
                doc.add_paragraph(line)
        buf = io.BytesIO()
        doc.save(buf)
        return Response(buf.getvalue(),
                        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        headers={"Content-Disposition": f'attachment; filename="export-{safe_title}.docx"'})

    elif fmt == "csv":
        import csv
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["AuRAG Export - " + title])
        writer.writerow([])
        for line in text.split("\n"):
            writer.writerow([line])
        return Response(buf.getvalue(), media_type="text/csv",
                                 headers={"Content-Disposition": f'attachment; filename="export-{safe_title}.csv"'})

    elif fmt in ["txt", "text"]:
        buf = f"AuRAG Export - {title}\n"
        buf += f"Generated at: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n"
        buf += "=" * 50 + "\n\n"
        buf += text
        return Response(buf, media_type="text/plain",
                                 headers={"Content-Disposition": f'attachment; filename="export-{safe_title}.txt"'})

    raise HTTPException(status_code=400, detail=f"Unsupported format: {fmt}")


@app.delete("/api/meetings/{meeting_id}/questions/{index}")
def delete_meeting_question(
    meeting_id: str,
    index: int,
    current_user: dict = Depends(get_current_user),
) -> dict[str, str]:
    meeting = SERVICE.get_meeting(meeting_id, user_id=_uid(current_user))
    if meeting is None:
        raise HTTPException(status_code=404, detail="Meeting not found")
    
    questions = meeting.get("questions", [])
    if index < 0 or index >= len(questions):
        raise HTTPException(status_code=400, detail="Invalid index")
    
    # Remove the single question at that index
    questions.pop(index)
    meeting["questions"] = questions
    SERVICE.save_meeting(meeting)
    return {"status": "success", "message": f"Deleted question at index {index}"}


@app.delete("/api/meetings/{meeting_id}")
def delete_meeting(meeting_id: str, current_user: dict = Depends(get_current_user)) -> dict[str, str]:
    success = SERVICE.delete_meeting(meeting_id, user_id=_uid(current_user))
    if not success:
        raise HTTPException(status_code=404, detail="Meeting not found or unauthorized")
    return {"status": "success", "message": f"Successfully deleted meeting {meeting_id}"}


@app.on_event("shutdown")
def shutdown_event() -> None:
    SERVICE.close()
