import os
import uuid
import io
import time
import logging
from typing import Dict, Any, List
from datetime import datetime

import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
from pinecone import Pinecone, ServerlessSpec
from sklearn.metrics.pairwise import cosine_similarity

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from groq import Groq


# =============================
# LOGGING
# =============================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("rag-backend")


# =============================
# GLOBAL STATE
# =============================
CONVERSATIONS: Dict[str, List[dict]] = {}
RESUMES: Dict[str, Dict[str, Any]] = {}


# =============================
# FASTAPI SETUP
# =============================
app = FastAPI(title="Resume RAG Backend", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================
# MODEL + VECTOR DB
# =============================
model = SentenceTransformer("all-MiniLM-L6-v2")

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

INDEX_NAME = "resume-rag-index"
DIMENSION = 384

if INDEX_NAME not in [i.name for i in pc.list_indexes()]:
    pc.create_index(
        name=INDEX_NAME,
        dimension=DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(INDEX_NAME)


# =============================
# FILE HELPERS
# =============================
def extract_text(file_bytes: bytes, filename: str):
    if filename.lower().endswith(".pdf"):
        reader = PdfReader(io.BytesIO(file_bytes))
        return "\n".join([p.extract_text() or "" for p in reader.pages])
    return file_bytes.decode("utf-8", errors="ignore")


def chunk_text(text, max_words=220, overlap=40):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunks.append(" ".join(words[i:i+max_words]))
        i += max_words - overlap
    return chunks


# =============================
# PDF BUILDER
# =============================
def build_pdf(data):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    y = 800

    for line in [
        "Resume Match Report",
        f"Score: {data['match_score']}",
        f"Strengths: {', '.join(data['strengths'])}",
        f"Gaps: {', '.join(data['gaps'])}",
        f"Insights: {data['insights']}"
    ]:
        c.drawString(50, y, line)
        y -= 30

    c.save()
    buffer.seek(0)
    return buffer


# =============================
# REQUEST MODEL
# =============================
class MatchRequest(BaseModel):
    resume_id: str
    jd_id: str


# =============================
# HEALTH
# =============================
@app.get("/")
def home():
    return {"status": "ok"}


# =============================
# UPLOAD RESUME
# =============================
@app.post("/upload_resume")
async def upload_resume(file: UploadFile = File(...)):
    start = time.time()

    raw = await file.read()
    text = extract_text(raw, file.filename)
    chunks = chunk_text(text)

    resume_id = str(uuid.uuid4())
    embs = model.encode(chunks)

    vectors = [
        {
            "id": f"{resume_id}_{i}",
            "values": e.tolist(),
            "metadata": {"resume_id": resume_id, "text": c}
        }
        for i, (c, e) in enumerate(zip(chunks, embs))
    ]

    index.upsert(vectors)

    logger.info(f"UPLOAD_RESUME {resume_id} chunks={len(chunks)} time={round(time.time()-start,2)}s")

    return {"resume_id": resume_id}


# =============================
# UPLOAD JD
# =============================
@app.post("/upload_jd")
async def upload_jd(file: UploadFile = File(...)):
    raw = await file.read()
    text = extract_text(raw, file.filename)
    chunks = chunk_text(text)

    jd_id = str(uuid.uuid4())
    embs = model.encode(chunks)

    vectors = [
        {
            "id": f"{jd_id}_{i}",
            "values": e.tolist(),
            "metadata": {"jd_id": jd_id, "text": c}
        }
        for i, (c, e) in enumerate(zip(chunks, embs))
    ]

    index.upsert(vectors)

    return {"jd_id": jd_id}


# =============================
# MATCH (AI + LIGHT EXPLAINABILITY)
# =============================
@app.post("/match")
async def match(payload: MatchRequest):
    start = time.time()

    # ----------------------------
    # Fetch Resume + JD Vectors
    # ----------------------------
    r = index.query(
        vector=[0]*DIMENSION,
        filter={"resume_id": payload.resume_id},
        top_k=50,
        include_metadata=True,
        include_values=True
    )

    j = index.query(
        vector=[0]*DIMENSION,
        filter={"jd_id": payload.jd_id},
        top_k=50,
        include_metadata=True,
        include_values=True
    )

    if not r.matches:
        raise HTTPException(404, "Resume not found")

    if not j.matches:
        raise HTTPException(404, "JD not found")

    # ----------------------------
    # Embedding Similarity (CORE AI SCORE)
    # ----------------------------
    rv = np.array([m.values for m in r.matches])
    jv = np.array([m.values for m in j.matches])

    score = float(cosine_similarity(rv, jv).mean()) * 100
    final_score = round(score, 2)

    # ----------------------------
    # LIGHT EXPLAINABILITY (Skill Hint Scan)
    # ----------------------------
    SKILL_HINTS = [
        "python", "java", "spring", "spring boot",
        "fastapi", "django", "react", "node",
        "sql", "mysql", "postgres", "mongodb",
        "aws", "docker", "kubernetes", "api",
        "rest", "microservices", "git", "cloud"
    ]

    resume_text = " ".join([m.metadata["text"] for m in r.matches]).lower()
    jd_text = " ".join([m.metadata["text"] for m in j.matches]).lower()

    jd_skills = [s for s in SKILL_HINTS if s in jd_text]

    strengths = [s for s in jd_skills if s in resume_text][:5]
    gaps = [s for s in jd_skills if s not in resume_text][:5]

    # ----------------------------
    # Insight (Simple + Clean)
    # ----------------------------
    if strengths:
        insight = f"Good alignment with {', '.join(strengths[:3])}."
    else:
        insight = "Semantic match found, but no strong skill keyword overlap detected."

    if gaps:
        insight += f" Consider improving exposure to {', '.join(gaps[:2])}."

    # ----------------------------
    # Final Response
    # ----------------------------
    result = {
        "match_score": f"{final_score}%",
        "strengths": strengths if strengths else ["Semantic alignment detected"],
        "gaps": gaps,
        "insights": insight
    }

    logger.info(
        f"MATCH score={final_score}% strengths={len(strengths)} gaps={len(gaps)} "
        f"time={round(time.time()-start,2)}s"
    )

    return result


# =============================
# PDF
# =============================
@app.post("/generate_report")
async def report(payload: MatchRequest):
    result = await match(payload)
    pdf = build_pdf(result)
    return StreamingResponse(pdf, media_type="application/pdf")


# =============================
# CHAT
# =============================
@app.post("/chat")
async def chat(payload: dict):
    q = payload["question"]
    resume_id = payload["resume_id"]

    emb = model.encode(q).tolist()

    res = index.query(
        vector=emb,
        filter={"resume_id": resume_id},
        top_k=5,
        include_metadata=True
    )

    context = "\n".join([m.metadata["text"] for m in res.matches])

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    resp = client.chat.completions.create(
        model="llama-3.1-8b-instant",
messages=[
{
"role": "user",
"content": f"""
You are an AI recruiter assistant helping evaluate a candidate.

STRICT RULES:
- Answer ONLY using the resume context provided.
- Do NOT use outside knowledge.
- If the answer is not clearly present, reply exactly:
  "Not found in resume."
- Keep answers concise and factual.

RESUME CONTEXT:
{context}

QUESTION:
{q}

FINAL ANSWER:
"""
}
]

    )

    return {"answer": resp.choices[0].message.content}
