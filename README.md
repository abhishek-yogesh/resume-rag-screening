# Resume RAG — resume-rag-fastapi

AI-powered Resume Screening using a Retrieval-Augmented Generation (RAG) pipeline with:
- FastAPI backend  
- React + Vite frontend  
- SentenceTransformers for embeddings  
- ChromaDB / Pinecone for vector search  
- Groq LLaMA 3 for scoring & explanation  

This system extracts text from resumes & JDs → chunks → embeds → retrieves similar segments → uses LLM reasoning to compute a match score and insights.

---

## 🔖 Badges

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi)
![React](https://img.shields.io/badge/React-Frontend-61DAFB?logo=react)
![Vite](https://img.shields.io/badge/Vite-Build%20Tool-646CFF?logo=vite)
![Groq](https://img.shields.io/badge/LLM-Groq%20LLaMA%203-orange)
![ChromaDB](https://img.shields.io/badge/VectorDB-Chroma-9cf)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🖼 System Architecture

<p align="center">
  <img src="./assets/Architecture.png" alt="Architecture Diagram" width="850">
</p>

---

## 📑 Table of Contents

- [Features](#-features)
- [Project Structure](#-project-structure)
- [Local Setup](#-local-setup)
  - [Backend Setup](#backend-fastapi)
  - [Frontend Setup](#frontend-react--vite)
- [API Documentation](#-api-documentation)
- [Sample Files](#-sample-files)
- [Deployment](#-deployment)
- [Environment Variables](#-environment-example)
- [License](#-license)

---

## 🚀 Features

- Upload Resume (PDF/TXT)
- Upload Job Description
- Automatic text extraction → chunking → embedding
- Vector similarity search (Chroma or Pinecone)
- LLM scoring & explanation (Groq LLaMA 3)
- Highlights matched & missing skills
- Optional Q&A for deeper insights

---

## 📂 Project Structure

    resume-rag-fastapi/
    ├── backend/
    │   ├── app/
    │   ├── main.py
    │   ├── requirements.txt
    │   └── .env.example
    │
    ├── frontend/
    │   ├── src/
    │   ├── public/
    │   ├── package.json
    │   └── vite.config.js
    │
    ├── assets/
    │   └── Architecture.png
    │
    ├── sample_files/
    │   ├── sample_resume_1.txt
    │   ├── sample_resume_2.txt
    │   ├── sample_jd_1.txt
    │   └── sample_jd_2.txt
    │
    ├── .gitignore
    └── README.md

---

## ⚙️ Local Setup

### Backend (FastAPI)

```bash
cd backend
python -m venv venv
venv\Scripts\Activate.ps1      # Windows PowerShell
pip install -r requirements.txt

uvicorn main:app --reload --port 8000
```

Backend UI:

- API root → http://localhost:8000  
- Swagger Docs → http://localhost:8000/docs  

---

### Frontend (React + Vite)

```bash
cd frontend
npm install
npm run dev
```

Open frontend:  
http://localhost:5173

---

## 📘 API Documentation

### POST /upload_resume

```bash
file=@resume.pdf
```

**Response:**
```json
{
  "status": "success",
  "resume_id": "uuid",
  "filename": "resume.pdf"
}
```

---

### POST /upload_jd

```bash
file=@jd.txt
```

**Response:**
```json
{
  "status": "success",
  "jd_id": "uuid",
  "filename": "jd.txt"
}
```

---

### POST /match-score

**Request:**
```json
{
  "resume_id": "uuid",
  "jd_id": "uuid"
}
```

**Response:**
```json
{
  "match_score": 82.4,
  "highlights": ["Strong React skills", "Missing AWS"],
  "explanation": "Based on retrieved context..."
}
```

---

### POST /query

**Request:**
```json
{
  "question": "What skills are missing?",
  "resume_id": "uuid",
  "jd_id": "uuid"
}
```

**Response:**
```json
{
  "answer": "The candidate lacks AWS deployment experience."
}
```

---

## 🧪 Sample Files

Located in `sample_files/`:

- sample_resume_1.txt  
- sample_resume_2.txt  
- sample_jd_1.txt  
- sample_jd_2.txt  

---

## 🚀 Deployment

### Backend (Render / Railway / EC2)

Environment variables:

```
GROQ_API_KEY=
PINECONE_API_KEY=
VECTOR_STORE=chroma
```

### Start command:

```bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```

---

### Frontend (Vercel / Netlify)

```
VITE_BACKEND_URL=https://your-backend-url
```

---

## 🧩 Environment Example

```
GROQ_API_KEY=
PINECONE_API_KEY=
VECTOR_STORE=chroma
VITE_BACKEND_URL=http://localhost:8000
```

---

## 📄 License

MIT License.

---

## 👤 Author

**Abhishek Yogesh**
