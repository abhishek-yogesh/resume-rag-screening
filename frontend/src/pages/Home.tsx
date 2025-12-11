// src/pages/Home.tsx
import React, { useState } from "react";
import UploadResume from "../components/UploadResume";
import UploadJD from "../components/UploadJD";
import MatchScore from "../components/MatchScore";
import Chat from "../components/Chat";
import "../App.css";

const Home: React.FC = () => {
  const [resumeId, setResumeId] = useState<string | null>(null);
  const [jdId, setJdId] = useState<string | null>(null);

  return (
    <div className="container">

      <div className="card">
        <UploadResume onUploaded={(id) => setResumeId(id)} />
      </div>

      <div className="card">
        <UploadJD onUploaded={(id) => setJdId(id)} />
      </div>

      {resumeId && jdId && (
        <div className="card">
          <MatchScore resumeId={resumeId} jdId={jdId} />
        </div>
      )}

      <div className="card">
        <Chat resumeId={resumeId} />
      </div>

    </div>
  );
};

export default Home;
