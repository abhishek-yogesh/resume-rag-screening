# Resume RAG — resume-rag-fastapi

AI-powered Resume Screening (RAG) using FastAPI backend, React frontend, SentenceTransformers for embeddings, Chroma/Pinecone vector stores, and Groq LLaMA 3 for reasoning and match scoring.

---

## 📌 System Architecture

![Architecture Diagram](./assets/Architecture.png)

> `/upload_resume` and `/upload_jd` perform preprocessing (text → chunk → embed).  
> `/match-score` only retrieves stored data → vector search → Groq LLaMA analysis.

---

## 🚀 Features
- Upload Resume (PDF/TXT)
- Upload Job Description
- Embedding + similarity search
- LLM (Groq LLaMA 3) scoring
- Highlights + missing skills + explanation
- Optional Chat/Q&A

---

## 📂 Project Structure
resume-rag-fastapi/
├── backend/
├── frontend/
├── assets/
│ └── architecture.png
├── sample_files/
│ ├── sample_resume_1.txt
│ ├── sample_resume_2.txt
│ ├── sample_jd_1.txt
│ └── sample_jd_2.txt
└── README.md



---

## ⚙️ Local Setup

### Backend (FastAPI)
```bash
cd backend
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
Open:
http://localhost:8000
Swagger:
http://localhost:8000/docs

Frontend (Vite React)
bash
Copy code
cd frontend
npm install
npm run dev
🔌 API Documentation
POST /upload_resume
Upload resume file → process → chunk → embed → store

Response

json
Copy code
{ "status":"success", "resume_id":"<uuid>" }
POST /upload_jd
Upload JD text → process → store

Response

json
Copy code
{ "status":"success", "jd_id":"<uuid>" }
POST /match-score
RAG pipeline → similarity search → Groq LLaMA reasoning

Response

json
Copy code
{
  "match_score": 82.4,
  "highlights": ["Strong React skills", "Missing AWS"],
  "explanation": "..."
}
POST /query (optional Q&A)
Provide follow-up questions about resume/JD.

🧪 Sample Files
Use the files in sample_files/ for testing.

🚀 Deployment
Backend → Render / Railway

Frontend → Vercel

Env vars:

GROQ_API_KEY

PINECONE_API_KEY

VITE_BACKEND_URL

🧾 License
MIT

