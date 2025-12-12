# Backend (FastAPI)

## Run locally
python -m venv venv
venv\Scripts\activate  (Windows)
source venv/bin/activate (Mac/Linux)

pip install -r requirements.txt
uvicorn main:app --reload --port 8000

Swagger UI: http://localhost:8000/docs
