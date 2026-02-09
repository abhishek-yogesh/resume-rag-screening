import os
import uuid
import io
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


# ----------------------------
# GLOBAL STATE
# ----------------------------
CONVERSATIONS: Dict[str, List[dict]] = {}
RESUMES: Dict[str, Dict[str, Any]] = {}


# ----------------------------
# FASTAPI SETUP
# ----------------------------
app = FastAPI(title="Resume RAG Backend", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("🔥 Backend loaded:", os.path.abspath(__file__))


# ----------------------------
# MODEL & VECTOR DB SETUP
# ----------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY not set")

pc = Pinecone(api_key=PINECONE_API_KEY)

INDEX_NAME = "resume-rag-index"
DIMENSION = 384

existing_indexes = [i.name for i in pc.list_indexes()]
if INDEX_NAME not in existing_indexes:
    pc.create_index(
        name=INDEX_NAME,
        dimension=DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(INDEX_NAME)


# ----------------------------
# FILE PROCESSING
# ----------------------------
def extract_text_from_file(file_bytes: bytes, filename: str) -> str:
    if filename.lower().endswith(".pdf"):
        reader = PdfReader(io.BytesIO(file_bytes))
        pages = [page.extract_text() or "" for page in reader.pages]
        text = "\n".join(pages).strip()
    else:
        text = file_bytes.decode("utf-8", errors="ignore").strip()

    if not text:
        raise HTTPException(400, "No text extracted")

    return text


def chunk_text(text: str, max_words=220, overlap=40) -> List[str]:
    words = text.split()
    if len(words) <= max_words:
        return [" ".join(words)]

    chunks = []
    i = 0
    while i < len(words):
        chunks.append(" ".join(words[i:i + max_words]))
        if i + max_words >= len(words):
            break
        i = i + max_words - overlap

    return chunks


# ----------------------------
# PDF GENERATION
# ----------------------------
def build_match_pdf(match_data: dict) -> io.BytesIO:
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    y = height - 50

    def draw(text: str):
        nonlocal y
        c.drawString(50, y, text)
        y -= 18
        if y < 50:
            c.showPage()
            y = height - 50

    c.setFont("Helvetica-Bold", 16)
    draw("Resume – Job Description Match Report")
    y -= 10

    c.setFont("Helvetica", 11)
    draw(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    y -= 20

    c.setFont("Helvetica-Bold", 12)
    draw(f"Final Match Score: {match_data['match_score']}")
    y -= 15

    c.setFont("Helvetica-Bold", 12)
    draw("Category Scores:")
    c.setFont("Helvetica", 11)
    for k, v in match_data.get("category_scores", {}).items():
        draw(f"- {k}: {v}%")
    y -= 15

    c.setFont("Helvetica-Bold", 12)
    draw("Strengths:")
    c.setFont("Helvetica", 11)
    for s in match_data.get("strengths", []):
        draw(f"- {s}")
    y -= 15

    c.setFont("Helvetica-Bold", 12)
    draw("Gaps:")
    c.setFont("Helvetica", 11)
    for g in match_data.get("gaps", []):
        draw(f"- {g}")
    y -= 15

    c.setFont("Helvetica-Bold", 12)
    draw("Insights:")
    c.setFont("Helvetica", 11)
    draw(match_data.get("insights", ""))

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer


# ----------------------------
# HEALTH
# ----------------------------
@app.get("/")
def home():
    return {"status": "ok"}


# ----------------------------
# UPLOAD RESUME
# ----------------------------
@app.post("/upload_resume")
async def upload_resume(file: UploadFile = File(...)):
    raw = await file.read()
    text = extract_text_from_file(raw, file.filename)
    chunks = chunk_text(text)

    resume_id = str(uuid.uuid4())
    embeddings = model.encode(chunks, batch_size=8)

    vectors = [
        {
            "id": f"{resume_id}_chunk_{i}",
            "values": emb.tolist(),
            "metadata": {"resume_id": resume_id, "text": chunk}
        }
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings))
    ]

    index.upsert(vectors)
    RESUMES[resume_id] = {"text": text}

    return {
        "status": "success",
        "resume_id": resume_id,
        "chunks": len(chunks)
    }


# ----------------------------
# UPLOAD JD
# ----------------------------
@app.post("/upload_jd")
async def upload_jd(file: UploadFile = File(...)):
    raw = await file.read()
    text = extract_text_from_file(raw, file.filename)
    chunks = chunk_text(text)

    jd_id = str(uuid.uuid4())
    embeddings = model.encode(chunks, batch_size=8)

    vectors = [
        {
            "id": f"{jd_id}_chunk_{i}",
            "values": emb.tolist(),
            "metadata": {"jd_id": jd_id, "text": chunk}
        }
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings))
    ]

    index.upsert(vectors)

    return {
        "status": "success",
        "jd_id": jd_id,
        "chunks": len(chunks)
    }

# ----------------------------
# SKILL DICTIONARIES
# ----------------------------
TECHNICAL_SKILLS = [
    "python", "java", "c++", "javascript", "react",
    "angular", "node", "fastapi", "django", "spring",
    "spring boot", "api", "rest api", "microservices",
    "sql", "mysql", "postgres", "mongodb", "redis",
    "docker", "kubernetes", "aws", "azure", "gcp",
    "cloud", "git", "github", "ci/cd"
]

SOFT_SKILLS = [
    "communication", "leadership", "problem solving",
    "teamwork", "collaboration", "analytical thinking"
]

# ----------------------------
# MATCH REQUEST MODEL
# ----------------------------
class MatchRequest(BaseModel):
    resume_id: str
    jd_id: str

def rerank_resume_chunks(resume_vecs, jd_vecs, top_k=15):

    if len(resume_vecs) == 0 or len(jd_vecs) == 0:
        return np.arange(len(resume_vecs)), None

    # ⭐ Production Safety
    top_k = min(top_k, len(resume_vecs))

    jd_mean_vec = jd_vecs.mean(axis=0).reshape(1, -1)

    scores = cosine_similarity(resume_vecs, jd_mean_vec).flatten()

    top_indices = np.argsort(scores)[::-1][:top_k]

    return top_indices, scores

# ----------------------------
# MATCH ANALYSIS
# ----------------------------
@app.post("/match")
async def match_resumes(payload: MatchRequest):

    resume_id = payload.resume_id
    jd_id = payload.jd_id

    # --- Query Resume ---
    resume_results = index.query(
        vector=[0] * DIMENSION,
        filter={"resume_id": resume_id},
        top_k=50,
        include_metadata=True,
        include_values=True,
    )

    resume_chunks = [m.metadata["text"] for m in resume_results.matches]
    resume_vectors = [m.values for m in resume_results.matches]

    if not resume_vectors:
        raise HTTPException(404, "Resume not found in vector DB")

    # --- Query JD ---
    jd_results = index.query(
        vector=[0] * DIMENSION,
        filter={"jd_id": jd_id},
        top_k=50,
        include_metadata=True,
        include_values=True,
    )

    jd_chunks = [m.metadata["text"] for m in jd_results.matches]
    jd_vectors = [m.values for m in jd_results.matches]

    if not jd_vectors:
        raise HTTPException(404, "JD not found in vector DB")

   # 1️⃣ Embedding similarity (WITH RERANK)

    resume_vecs = np.array(resume_vectors)
    jd_vecs = np.array(jd_vectors)

    # ---- RERANK STAGE ----
    top_indices, rerank_scores = rerank_resume_chunks(
        resume_vecs,
        jd_vecs,
        top_k=15
    )

    filtered_resume_vecs = resume_vecs[top_indices]
    filtered_resume_chunks = [resume_chunks[i] for i in top_indices]

    # ---- FINAL SIMILARITY ----
    sim_matrix = cosine_similarity(filtered_resume_vecs, jd_vecs)

    embedding_score = round(float(sim_matrix.mean()) * 100, 2)

    # 2️⃣ Preprocess text
    resume_text = " ".join(filtered_resume_chunks).lower()
    jd_text = " ".join(jd_chunks).lower()

    # 3️⃣ Skill matching
    jd_skills = [s for s in TECHNICAL_SKILLS + SOFT_SKILLS if s in jd_text]
    strengths = [s for s in jd_skills if s in resume_text]
    gaps = [s for s in jd_skills if s not in resume_text]

    weaknesses = []
    if any(x in gaps for x in ["docker", "kubernetes", "aws", "gcp"]):
         weaknesses.append("Missing DevOps/Cloud experience")

    if "ci/cd" in gaps:
        weaknesses.append("Lack of CI/CD exposure")


    skills_score = round(
            (len(strengths) / len(jd_skills) * 100), 2
        ) if jd_skills else 0

        # 4️⃣ Tools matching
    tools = ["docker", "kubernetes", "k8s", "aws", "gcp", "azure", "jenkins"]
    jd_tools = [t for t in tools if t in jd_text]
    resume_tools = [t for t in tools if t in resume_text]
    matched_tools = [t for t in jd_tools if t in resume_tools]

    tools_score = round(
        (len(matched_tools) / len(jd_tools)) * 100, 2
        ) if jd_tools else 0

    # 5️⃣ Experience matching
    import re

    def extract_years(text: str):
            match = re.search(r"(\d+)\s*(\+)?\s*(years?|yrs?|year)", text)
            return int(match.group(1)) if match else 0

    resume_years = extract_years(resume_text)
    jd_years = extract_years(jd_text)

    experience_score = (
            min(round((resume_years / jd_years) * 100, 2), 100)
            if jd_years else 0
        )

        # 6️⃣ JD keyword coverage
    jd_tokens = [t for t in jd_text.split() if len(t) > 2]
    resume_tokens = set([t for t in resume_text.split() if len(t) > 2])

    covered = len([t for t in jd_tokens if t in resume_tokens])
    jd_keyword_coverage = round(
            (covered / len(jd_tokens) * 100), 2
        ) if jd_tokens else 0

        # 7️⃣ Final weighted score
    final_score = round(
            0.40 * embedding_score +
            0.30 * skills_score +
            0.15 * tools_score +
            0.10 * experience_score +
            0.05 * jd_keyword_coverage,
            2
        )

        # 8️⃣ Insight
    insight = (
            f"Strong alignment with {', '.join(strengths[:3]) if strengths else 'some skills'} "
            f"but missing {', '.join(gaps[:3]) if gaps else 'no major gaps'}. "
            f"Final weighted match score: {final_score}%."
        )

        # 9️⃣ Response
    return {
    "match_score": f"{final_score}%",
    "strengths": strengths,
    "gaps": gaps,
    "weaknesses": weaknesses,
    "insights": insight,
    "category_scores": {
        "Skills Match": skills_score,
        "Tools Match": tools_score,
        "Experience Match": experience_score,
        "JD Keyword Coverage": jd_keyword_coverage
    }
}




# ----------------------------
# PDF REPORT ENDPOINT
# ----------------------------
@app.post("/generate_report")
async def generate_report(payload: MatchRequest):
    match_result = await match_resumes(payload)
    pdf_buffer = build_match_pdf(match_result)

    return StreamingResponse(
        pdf_buffer,
        media_type="application/pdf",
        headers={
            "Content-Disposition": "attachment; filename=resume_match_report.pdf"
        },
    )


# ----------------------------
# RAG CHAT
# ----------------------------
@app.post("/chat")
async def rag_chat(payload: dict):

    question = payload.get("question")
    resume_id = payload.get("resume_id")

    if not question or not resume_id:
        raise HTTPException(400, "Missing question or resume_id")

    if resume_id not in CONVERSATIONS:
        CONVERSATIONS[resume_id] = []

    CONVERSATIONS[resume_id].append({"role": "user", "content": question})

    # -----------------------
    # Embed Question
    # -----------------------
    question_embedding = model.encode(question).tolist()

    # -----------------------
    # Fetch More Candidates ⭐
    # -----------------------
    results = index.query(
        vector=question_embedding,
        filter={"resume_id": resume_id},
        top_k=20,
        include_metadata=True,
        include_values=True
    )

    chunks = [m.metadata["text"] for m in results.matches]
    vectors = [m.values for m in results.matches]

    if not vectors:
        return {"answer": "No resume context found."}

    # -----------------------
    # ⭐ RERANK
    # -----------------------
    resume_vecs = np.array(vectors)
    question_vec = np.array(question_embedding).reshape(1, -1)

    top_indices, _ = rerank_resume_chunks(
        resume_vecs,
        question_vec,
        top_k=5
    )

    filtered_chunks = [chunks[i] for i in top_indices]

    # -----------------------
    # Build Context
    # -----------------------
    context = "\n".join(filtered_chunks)

    prompt = f"""
You are an assistant helping answer questions about a candidate resume.

Resume Context:
{context}

Question:
{question}

Answer clearly using only resume context.
"""

    # -----------------------
    # ⭐ LLM CALL
    # -----------------------
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "user", "content": prompt}
        ],
    )

    answer = response.choices[0].message.content

    CONVERSATIONS[resume_id].append({"role": "ai", "content": answer})

    return {"answer": answer}
