import React, { useState } from "react";
import { uploadResume } from "../api/api";

interface UploadResumeProps {
  onUploaded: (resumeId: string) => void;
}

const UploadResume: React.FC<UploadResumeProps> = ({ onUploaded }) => {
  const [file, setFile] = useState<File | null>(null);
  const [status, setStatus] = useState("");
  const [resumeId, setResumeId] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const handleUpload = async () => {
    if (!file) {
      setStatus("Please choose a resume file.");
      return;
    }

    try {
      setLoading(true);
      setStatus("Uploading resume...");

      const res = await uploadResume(file);

      if (res.resume_id) {
        setResumeId(res.resume_id);
        onUploaded(res.resume_id);
        setStatus("Resume uploaded successfully!");
      } else {
        setStatus("Upload succeeded, but backend returned no resume_id.");
      }
    } catch (err) {
      console.error(err);
      setStatus("Failed to upload resume.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="card">
      <h3>📄 Upload Resume</h3>

      <input
        type="file"
        accept=".pdf,.txt"
        onChange={(e) => setFile(e.target.files?.[0] ?? null)}
      />

      <button className="btn" onClick={handleUpload} disabled={loading}>
        {loading ? (
          <div className="spinner-inline"></div>
        ) : (
          "Upload Resume"
        )}
      </button>

      {status && <p className="status-text">{status}</p>}

      {resumeId && (
        <p className="small-muted">
          Resume ID: <code>{resumeId}</code>
        </p>
      )}
    </div>
  );
};

export default UploadResume;
