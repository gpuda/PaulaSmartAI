import os
import json
import math
from typing import List, Dict, Any, Optional, Tuple, Literal

from dotenv import load_dotenv
from openai import OpenAI

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

from app.agent import get_agent

# LangChain message types (da LangGraph agent pouzdanije koristi toolove)
from langchain_core.messages import HumanMessage, AIMessage


# =========================
# ENV + OPENAI
# =========================
# agent.py ucitava api/.env; ovdje ostavljamo load_dotenv() radi OPENAI_API_KEY
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")

# Embeddings model for RAG
EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

# PDF path for RAG
RAG_PDF_PATH = os.getenv("RAG_PDF_PATH", "data/50Prirucnik-za-vjezbe_Davidovic-Cvetko.pdf")

# Chunking params
CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "900"))        # chars
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "150"))  # chars
TOP_K = int(os.getenv("RAG_TOP_K", "5"))


# =========================
# FASTAPI APP
# =========================
app = FastAPI(title="test_app_agentic_rag")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# MODELS
# =========================
class ChatRequest(BaseModel):
    message: str
    history: Optional[List[Dict[str, str]]] = None


class UiChatRequest(BaseModel):
    message: str
    mode: Literal["normal", "web", "rag"] = "normal"
    top_k: Optional[int] = None  # koristi se samo za rag
    history: Optional[List[Dict[str, str]]] = None


class RagChatRequest(BaseModel):
    message: str
    top_k: Optional[int] = None
    history: Optional[List[Dict[str, str]]] = None


# =========================
# SIMPLE RAG (in-memory)
# =========================
RAG_READY = False
RAG_ERROR: Optional[str] = None

_chunks: List[Dict[str, Any]] = []
_embeds: List[List[float]] = []
_embed_norms: List[float] = []


def _safe_norm(vec: List[float]) -> float:
    s = 0.0
    for x in vec:
        s += x * x
    return math.sqrt(s) if s > 0 else 1e-12


def _cosine_sim(a: List[float], b: List[float], na: float, nb: float) -> float:
    dot = 0.0
    for i in range(len(a)):
        dot += a[i] * b[i]
    return dot / (na * nb)


def _chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    text = text.strip()
    if not text:
        return []
    out = []
    start = 0
    step = max(1, chunk_size - overlap)
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunk = text[start:end].strip()
        if chunk:
            out.append(chunk)
        start += step
    return out


def _read_pdf_text(path: str) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Returns: full_text, per_page_meta list with page_text
    """
    try:
        from pypdf import PdfReader
    except Exception:
        try:
            from PyPDF2 import PdfReader  # type: ignore
        except Exception as e:
            raise RuntimeError("Nedostaje pypdf/PyPDF2. Instaliraj: pip install pypdf") from e

    reader = PdfReader(path)
    per_page = []
    all_text_parts = []
    for i, page in enumerate(reader.pages):
        t = page.extract_text() or ""
        t = t.replace("\r", "\n")
        per_page.append({"page": i + 1, "text": t})
        all_text_parts.append(f"\n\n--- PAGE {i+1} ---\n{t}")
    return "\n".join(all_text_parts), per_page


def _embed(texts: List[str]) -> List[List[float]]:
    resp = client.embeddings.create(
        model=EMBED_MODEL,
        input=texts,
    )
    return [d.embedding for d in resp.data]


def _build_rag_index() -> None:
    global RAG_READY, RAG_ERROR, _chunks, _embeds, _embed_norms

    if not os.path.exists(RAG_PDF_PATH):
        RAG_READY = False
        RAG_ERROR = (
            f"PDF nije pronaden na putanji: {RAG_PDF_PATH}. "
            "Postavi RAG_PDF_PATH ili kopiraj PDF u data/."
        )
        return

    try:
        _, per_page = _read_pdf_text(RAG_PDF_PATH)

        chunks = []
        for p in per_page:
            page_no = p["page"]
            page_text = (p["text"] or "").strip()
            if not page_text:
                continue
            page_chunks = _chunk_text(page_text, CHUNK_SIZE, CHUNK_OVERLAP)
            for idx, ch in enumerate(page_chunks):
                chunks.append(
                    {
                        "id": f"p{page_no}_c{idx}",
                        "page": page_no,
                        "text": ch,
                    }
                )

        if not chunks:
            RAG_READY = False
            RAG_ERROR = "PDF je ucitan, ali nisam izvukao tekst (mozda je sken)."
            return

        embeds = []
        BATCH = 64
        for i in range(0, len(chunks), BATCH):
            batch_texts = [c["text"] for c in chunks[i : i + BATCH]]
            embeds.extend(_embed(batch_texts))

        norms = [_safe_norm(v) for v in embeds]

        _chunks = chunks
        _embeds = embeds
        _embed_norms = norms
        RAG_READY = True
        RAG_ERROR = None

    except Exception as e:
        RAG_READY = False
        RAG_ERROR = f"RAG init error: {type(e).__name__}: {e}"


def _retrieve(query: str, top_k: int) -> List[Dict[str, Any]]:
    if not RAG_READY:
        return []

    q_emb = _embed([query])[0]
    q_norm = _safe_norm(q_emb)

    scored = []
    for i in range(len(_embeds)):
        sim = _cosine_sim(q_emb, _embeds[i], q_norm, _embed_norms[i])
        scored.append((sim, i))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[: max(1, top_k)]

    results = []
    for sim, idx in top:
        c = _chunks[idx]
        results.append(
            {
                "score": float(sim),
                "page": c["page"],
                "id": c["id"],
                "text": c["text"],
            }
        )
    return results


def _rag_answer(user_message: str, contexts: List[Dict[str, Any]]) -> str:
    ctx_lines = []
    for i, c in enumerate(contexts, start=1):
        ctx_lines.append(f"[Izvor {i} | str {c['page']}] {c['text']}")
    ctx_block = "\n\n".join(ctx_lines) if ctx_lines else "(Nema pronadenog konteksta.)"

    system = (
        "Ti si asistent za fiziologiju sporta i vjezbanja (fizioterapija). "
        "Odgovaraj kratko i jasno, na hrvatskom bez dijakritike. "
        "Ako informacija nije u izvorima, reci da nisi siguran i predlozi sto provjeriti."
    )

    prompt = (
        f"KORISNIKOVO PITANJE:\n{user_message}\n\n"
        f"IZVORI (iz PDF prirucnika):\n{ctx_block}\n\n"
        "ZADATAK:\n"
        "- Odgovori korisniku koristeci izvore.\n"
        "- Ako se pozivas na nesto iz izvora, spomeni stranicu (npr. 'str 34').\n"
    )

    resp = client.responses.create(
        model=MODEL,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    )
    return resp.output_text


# Build RAG index on startup
@app.on_event("startup")
def on_startup():
    _build_rag_index()


# =========================
# ROUTES
# =========================
@app.get("/health")
def health():
    return {
        "status": "ok",
        "rag_ready": RAG_READY,
        "rag_pdf_path": RAG_PDF_PATH,
        "rag_error": RAG_ERROR,
        "chunks": len(_chunks),
    }


@app.post("/chat")
def chat(req: ChatRequest):
    """
    Non-streaming agent chat (LangGraph). Agent trenutno ima web_search tool.
    """
    agent = get_agent()

    msgs: List[Any] = []
    if req.history:
        for h in req.history:
            role = (h.get("role") or "").lower()
            content = h.get("content") or ""
            if not content:
                continue
            if role == "user":
                msgs.append(HumanMessage(content=content))
            elif role == "assistant":
                msgs.append(AIMessage(content=content))

    msgs.append(HumanMessage(content=req.message))

    result = agent.invoke({"messages": msgs})
    msgs = result.get("messages", [])

    tool_output = ""
    for m in reversed(msgs):
        if getattr(m, "type", "") == "tool":
            tool_output = getattr(m, "content", "") or ""
            break

    tool_used = False
    tool_names: List[str] = []
    for m in msgs:
        if getattr(m, "type", "") == "tool":
            tool_used = True
            name = getattr(m, "name", None) or getattr(m, "tool", None) or ""
            if name:
                tool_names.append(str(name))

    final_text = ""
    if msgs:
        last = msgs[-1]
        if isinstance(last, dict):
            final_text = last.get("content", "") or ""
        else:
            final_text = getattr(last, "content", "") or ""

    payload = {
        "aiReply": (final_text or "").strip(),
        "tool_used": tool_used,
        "tool_names": tool_names,
        "tool_output": tool_output,
    }

    return Response(
        content=json.dumps(payload, ensure_ascii=False),
        media_type="application/json; charset=utf-8",
    )


@app.post("/chat_stream")
def chat_stream(req: ChatRequest):
    """
    Streaming chat (SSE): sends incremental deltas as they arrive.
    (Ovo je i dalje "stari" streaming bez LangGraph agenta.)
    """
    def event_generator():
        system_msg = {
            "role": "system",
            "content": "You are PaulaSmartAI, a friendly assistant for a 10-year-old girl in Croatia.",
        }

        history_messages: List[Dict[str, str]] = []
        if req.history:
            for h in req.history:
                role = (h.get("role") or "").lower()
                content = h.get("content") or ""
                if role not in ("user", "assistant") or not content:
                    continue
                history_messages.append({"role": role, "content": content})

        messages_payload: List[Dict[str, str]] = [system_msg, *history_messages, {"role": "user", "content": req.message}]

        stream = client.chat.completions.create(
            model=MODEL,
            messages=messages_payload,
            stream=True,
        )

        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                yield f"data: {json.dumps({'type': 'delta', 'delta': delta}, ensure_ascii=False)}\n\n"

        yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream; charset=utf-8",
    )


# -------- NEW: single endpoint za frontend toggle (normal/web/rag) --------
@app.post("/chat_router")
def chat_router(req: UiChatRequest):
    """
    Jedan endpoint za UI.
    - mode="normal": streaming SSE iz /chat_stream
    - mode="rag":    streaming SSE iz /rag_stream
    - mode="web":    non-stream agent odgovor iz /chat (LangGraph + web_search)
    """
    if req.mode == "web":
        # koristi agent (non-stream)
        return chat(ChatRequest(message=req.message, history=req.history))

    if req.mode == "rag":
        # koristi RAG streaming
        return rag_stream(
            RagChatRequest(message=req.message, top_k=req.top_k, history=req.history)
        )

    # default: normal streaming
    return chat_stream(ChatRequest(message=req.message, history=req.history))


# =========================
# RAG ROUTES
# =========================
@app.post("/rag_chat")
def rag_chat(req: RagChatRequest):
    """
    Non-streaming RAG chat: retrieves top_k chunks from PDF, then answers with citations to pages.
    """
    if not RAG_READY:
        payload = {
            "aiReply": "RAG nije spreman. Provjeri /health za detalje.",
            "rag_ready": False,
            "rag_error": RAG_ERROR,
        }
        return Response(
            content=json.dumps(payload, ensure_ascii=False),
            media_type="application/json; charset=utf-8",
        )

    top_k = req.top_k or TOP_K
    contexts = _retrieve(req.message, top_k=top_k)
    answer = _rag_answer(req.message, contexts)

    payload = {
        "aiReply": answer,
        "rag_ready": True,
        "sources": [{"page": c["page"], "score": c["score"], "id": c["id"]} for c in contexts],
    }
    return Response(
        content=json.dumps(payload, ensure_ascii=False),
        media_type="application/json; charset=utf-8",
    )


@app.post("/rag_stream")
def rag_stream(req: RagChatRequest):
    """
    Streaming RAG (SSE): emits deltas. At the end emits a 'sources' event.
    """
    def event_generator():
        if not RAG_READY:
            yield (
                "data: "
                + json.dumps(
                    {"type": "error", "message": "RAG nije spreman", "rag_error": RAG_ERROR},
                    ensure_ascii=False,
                )
                + "\n\n"
            )
            yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"
            return

        top_k = req.top_k or TOP_K
        contexts = _retrieve(req.message, top_k=top_k)

        ctx_lines = []
        for i, c in enumerate(contexts, start=1):
            ctx_lines.append(f"[Izvor {i} | str {c['page']}] {c['text']}")
        ctx_block = "\n\n".join(ctx_lines) if ctx_lines else "(Nema pronadenog konteksta.)"

        system = (
            "Ti si asistent za fiziologiju sporta i vjezbanja (fizioterapija). "
            "Odgovaraj kratko i jasno, na hrvatskom bez dijakritike. "
            "Ako informacija nije u izvorima, reci da nisi siguran i predlozi sto provjeriti."
        )

        history_block = ""
        if req.history:
            lines = []
            for h in req.history:
                role = (h.get("role") or "").lower()
                content = h.get("content") or ""
                if not content:
                    continue
                if role == "user":
                    prefix = "KORISNIK"
                elif role == "assistant":
                    prefix = "ASISTENT"
                else:
                    continue
                lines.append(f"{prefix}: {content}")
            if lines:
                history_block = "DOSADASNJI RAZGOVOR:\n" + "\n".join(lines) + "\n\n"

        prompt = (
            f"{history_block}KORISNIKOVO PITANJE:\n{req.message}\n\n"
            f"IZVORI (iz PDF prirucnika):\n{ctx_block}\n\n"
            "ZADATAK:\n"
            "- Odgovori korisniku koristeci izvore.\n"
            "- Ako se pozivas na nesto iz izvora, spomeni stranicu (npr. 'str 34').\n"
        )

        stream = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            stream=True,
        )

        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                yield f"data: {json.dumps({'type': 'delta', 'delta': delta}, ensure_ascii=False)}\n\n"

        yield (
            "data: "
            + json.dumps(
                {
                    "type": "sources",
                    "sources": [{"page": c["page"], "score": c["score"], "id": c["id"]} for c in contexts],
                },
                ensure_ascii=False,
            )
            + "\n\n"
        )
        yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream; charset=utf-8",
    )