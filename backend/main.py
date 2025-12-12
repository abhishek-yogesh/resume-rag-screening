import os
import uuid
import io
from typing import Dict, Any, List

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader

from pinecone import Pinecone, ServerlessSpec
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from groq import Groq

# Store conversation memory per resume
CONVERSATIONS: Dict[str, List[dict]] = {}

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


# ----------------------------
# MODEL & PINECONE SETUP
# ----------------------------
print("🔥 Backend loaded:", os.path.abspath(__file__))

model = SentenceTransformer("all-MiniLM-L6-v2")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY not set")

pc = Pinecone(api_key=PINECONE_API_KEY)

INDEX_NAME = "resume-rag-index"
DIMENSION = 384

# Create index if not exists
existing = [i.name for i in pc.list_indexes()]
if INDEX_NAME not in existing:
    pc.create_index(
        name=INDEX_NAME,
        dimension=DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(INDEX_NAME)

RESUMES: Dict[str, Dict[str, Any]] = {}


# ----------------------------
# FILE PROCESSING FUNCTIONS
# ----------------------------
def extract_text_from_file(file_bytes: bytes, filename: str) -> str:
    name = filename.lower()

    if name.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(file_bytes))
        pages = [page.extract_text() or "" for page in reader.pages]
        text = "\n".join(pages).strip()
    else:
        text = file_bytes.decode("utf-8", errors="ignore").strip()

    if not text:
        raise HTTPException(400, "No text extracted")
    return text


def chunk_text(text: str, max_words=220, overlap=40):
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
    embeddings = [e.tolist() for e in embeddings]

    vectors = [
        {
            "id": f"{resume_id}_chunk_{i}",
            "values": emb,
            "metadata": {"resume_id": resume_id, "text": chunk}
        }
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings))
    ]

    index.upsert(vectors=vectors)

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

    index.upsert(vectors=vectors)

    return {
        "status": "success",
        "jd_id": jd_id,
        "chunks": len(chunks)
    }


# ----------------------------
# SKILL DICTIONARY
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


class MatchRequest(BaseModel):
    resume_id: str
    jd_id: str


# ----------------------------
# MATCH ENDPOINT 
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

    # ------------------------------
    # 1️⃣ Embedding Similarity Score
    # ------------------------------
    resume_vecs = np.array(resume_vectors)
    jd_vecs = np.array(jd_vectors)
    sim_matrix = cosine_similarity(resume_vecs, jd_vecs)
    embedding_score = round(float(sim_matrix.mean()) * 100, 2)

    # ------------------------------
    # 2️⃣ Preprocess text
    # ------------------------------
    resume_text = " ".join(resume_chunks).lower()
    jd_text = " ".join(jd_chunks).lower()

    # ------------------------------
    # 3️⃣ Skill Matching
    # ------------------------------
    jd_skills = [s for s in TECHNICAL_SKILLS + SOFT_SKILLS if s in jd_text]
    strengths = [s for s in jd_skills if s in resume_text]
    gaps = [s for s in jd_skills if s not in resume_text]

    # --- Weakness messages ---
    weaknesses = []
    if any(x in gaps for x in ["docker", "kubernetes", "aws", "gcp"]):
        weaknesses.append("Missing DevOps/Cloud experience")
    if "ci/cd" in gaps:
        weaknesses.append("Lack of CI/CD exposure")

    skills_score = round((len(strengths) / len(jd_skills) * 100), 2) if jd_skills else 0

    # ------------------------------
    # 4️⃣ Tools Matching 
    # ------------------------------
    tools = ["docker", "kubernetes", "k8s", "aws", "gcp", "azure", "jenkins"]

    jd_tools = [t for t in tools if t in jd_text]
    resume_tools = [t for t in tools if t in resume_text]

    matched_tools = [t for t in jd_tools if t in resume_tools]

    tools_score = (
        round((len(matched_tools) / len(jd_tools)) * 100, 2) if jd_tools else 0
    )

    # ------------------------------
    # 5️⃣ Experience Matching 
    # ------------------------------
    import re

    def extract_years(text: str):
        match = re.search(r"(\d+)\s*(\+)?\s*(years?|yrs?|year)", text)
        return int(match.group(1)) if match else 0

    resume_years = extract_years(resume_text)
    jd_years = extract_years(jd_text)

    if jd_years == 0:
        experience_score = 0
    else:
        experience_score = min(round((resume_years / jd_years) * 100, 2), 100)

    # ------------------------------
    # 6️⃣ JD Keyword Coverage
    # ------------------------------
    jd_tokens = [t for t in jd_text.split() if len(t) > 2]
    resume_tokens = set([t for t in resume_text.split() if len(t) > 2])

    covered = len([t for t in jd_tokens if t in resume_tokens])
    jd_keyword_coverage = round((covered / len(jd_tokens) * 100), 2) if jd_tokens else 0

    # ------------------------------
    # 7️⃣ FINAL WEIGHTED SCORE 
    # ------------------------------
    final_score = round(
        0.40 * embedding_score +
        0.30 * skills_score +
        0.15 * tools_score +
        0.10 * experience_score +
        0.05 * jd_keyword_coverage,
        2
    )

    # ------------------------------
    # 8️⃣ Insight Message
    # ------------------------------
    insight = (
        f"Strong alignment with {', '.join(strengths[:3]) if strengths else 'some skills'} "
        f"but missing {', '.join(gaps[:3]) if gaps else 'no major gaps'}. "
        f"Final weighted match score: {final_score}%."
    )

    # ------------------------------
    # 9️⃣ Final Response
    # ------------------------------
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
# RAG CHAT 
# ----------------------------
@app.post("/chat")
async def rag_chat(payload: dict):
    question = payload.get("question")
    resume_id = payload.get("resume_id")

    if not question:
        raise HTTPException(status_code=400, detail="Missing 'question'")
    if not resume_id:
        raise HTTPException(status_code=400, detail="Missing 'resume_id'")

    # -------------------------
    # 1️⃣ Initialize memory
    # -------------------------
    if resume_id not in CONVERSATIONS:
        CONVERSATIONS[resume_id] = []

    # Save user question to memory
    CONVERSATIONS[resume_id].append({"role": "user", "content": question})

    # -------------------------
    # 2️⃣ Embed the question (real RAG step)``
    # -------------------------
    question_embedding = model.encode(question).tolist()

    results = index.query(
        vector=question_embedding,
        filter={"resume_id": resume_id},
        top_k=5,
        include_metadata=True
    )

    context_text = "\n\n".join(
        [m.metadata.get("text", "") for m in results.matches]
    ) or "No relevant resume text found."

    # -------------------------
    # 3️⃣ Build conversation prompt
    # -------------------------
    history_text = ""
    for msg in CONVERSATIONS[resume_id][-6:]:  # last 3 Q&A pairs
        prefix = "User" if msg["role"] == "user" else "AI"
        history_text += f"{prefix}: {msg['content']}\n"

    prompt = f"""
You are an intelligent AI resume assistant.
You will answer questions about the resume ONLY using the provided context.

RESUME CONTEXT:
{context_text}

CONVERSATION HISTORY:
{history_text}

USER QUESTION:
{question}

Give a helpful, concise answer. If the resume does not contain enough information, say so.
"""

    # -------------------------
    # 4️⃣ Call Groq LLM
    # -------------------------
    try:
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You analyze resumes and give helpful guidance."},
                {"role": "user", "content": prompt},
            ],
        )
        answer = response.choices[0].message.content
    except Exception as e:
        print("ERROR:", e)
        answer = "Sorry — failed to generate answer."

    # Save AI answer to memory
    CONVERSATIONS[resume_id].append({"role": "ai", "content": answer})

    return {"answer": answer}
