# Resume RAG (resume-rag-fastapi)

**Resume RAG** is an AI-powered resume screening system using a RAG pipeline: chunking → embeddings → vector search → LLM reasoning (Groq LLaMA 3).

---

## Quick Links
- Backend (FastAPI): `backend/`
- Frontend (React + Vite): `frontend/`
- Samples: `sample_files/`
- Assets (diagram): `assets/architecture.png`

---

## Architecture (short)
User → Frontend → Upload Resume/JD → Backend API (FastAPI)
→ (Resume Processor, JD Processor) → In-Memory Storage → Text Chunking → Embedding Model (SentenceTransformers) → Vector Store (ChromaDB / Pinecone) → LLM — Groq (LLaMA 3) → Final Output (Match Score + Highlights)

> Match-score runtime: `/match-score` reads In-Memory Storage → Vector Store → LLM → returns score + highlights (no re-chunking on match request).

Embed architecture diagram here:
```md
![Architecture](./assets/architecture.png)
