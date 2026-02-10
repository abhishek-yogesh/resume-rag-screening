// src/api/api.ts

export const BASE_URL = "http://127.0.0.1:8000";
export async function sendChat(resumeId: string, question: string) {
  const response = await fetch("http://127.0.0.1:8000/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ resume_id: resumeId, question }),
  });

  if (!response.ok) {
    throw new Error("Chat request failed");
  }

  return await response.json();
}


// ---------- TYPES ----------
export interface UploadResponse {
  status: string;
  resume_id?: string;
  jd_id?: string;
  filename: string;
  chunks_stored: number;
}

export interface MatchAnalysis {
  match_score: string;       // "31.4%"
  strengths: string[];
  gaps: string[];
  weaknesses: string[];
  insights: string;
}

// ---------- UPLOAD RESUME ----------
export async function uploadResume(file: File): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(`${BASE_URL}/upload_resume`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) throw new Error("Failed to upload resume");

  return (await res.json()) as UploadResponse;
}

// ---------- UPLOAD JD ----------
export async function uploadJD(file: File): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(`${BASE_URL}/upload_jd`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    console.error("JD upload failed:", await res.text());
    throw new Error("Failed to upload JD");
  }

  const data = (await res.json()) as UploadResponse;
  console.log("JD upload response:", data);
  return data;
}

// ---------- MATCH ANALYSIS ----------
export async function getMatchAnalysis(
  resume_id: string,
  jd_id: string
): Promise<MatchAnalysis> {
  const res = await fetch(`${BASE_URL}/match`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ resume_id, jd_id }),
  });

  if (!res.ok) throw new Error(await res.text());

  return (await res.json()) as MatchAnalysis;
}

// ---------- RAG CHAT ----------
export async function askQuestion(question: string, resumeId: string) {
  const res = await fetch(`${BASE_URL}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, resume_id: resumeId }),
  });

  const data = await res.json();
  return data;
}

export async function generatePDF(resumeId: string, jdId: string) {
  const res = await fetch(`${BASE_URL}/generate_report`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ resume_id: resumeId, jd_id: jdId }),
  });

  if (!res.ok) {
    throw new Error("Failed to generate PDF");
  }

  return await res.blob();
}
