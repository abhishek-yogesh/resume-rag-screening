// src/components/UploadJD.tsx
import React, { useState } from "react";
import { uploadJD } from "../api/api";

interface UploadJDProps {
  onUploaded: (jdId: string) => void;
}

const UploadJD: React.FC<UploadJDProps> = ({ onUploaded }) => {
  const [file, setFile] = useState<File | null>(null);
  const [status, setStatus] = useState<string>("");
  const [jdId, setJdId] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const handleUpload = async () => {
    if (!file) {
      setStatus("Please choose a JD file first.");
      return;
    }
    try {
      setLoading(true);
      setStatus("Uploading JD...");
      const res = await uploadJD(file);

      if (res.jd_id) {
        setJdId(res.jd_id);
        onUploaded(res.jd_id); // ✅ push up to parent
        setStatus(`Uploaded! JD ID: ${res.jd_id}`);
      } else {
        setStatus("Uploaded, but no jd_id returned from backend.");
      }
    } catch (err) {
      console.error(err);
      setStatus("Failed to upload JD.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="card">
      <h3>📃 Upload Job Description</h3>

      <input
        type="file"
        accept=".pdf,.txt"
        onChange={(e) => setFile(e.target.files?.[0] ?? null)}
      />

      <button onClick={handleUpload} disabled={loading}>
        {loading ? "Uploading..." : "Upload JD"}
      </button>

      {status && <p className="status-text">{status}</p>}
      {jdId && (
        <p className="small-muted">
          Current JD ID: <code>{jdId}</code>
        </p>
      )}
    </div>
  );
};

export default UploadJD;
