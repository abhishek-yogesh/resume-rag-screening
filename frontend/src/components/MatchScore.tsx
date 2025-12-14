import React, { useState, useEffect } from "react";
import { getMatchAnalysis, generatePDF } from "../api/api";
import "./MatchScore.css";

interface MatchScoreProps {
  resumeId: string | null;
  jdId: string | null;
}

const MatchScore: React.FC<MatchScoreProps> = ({ resumeId, jdId }) => {
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);

    //  Match score (number only)
  const getScoreValue = (score: string | null) => {
    if (!score) return null;
    const value = Number(score.replace("%", ""));
    return isNaN(value) ? null : value;
  };

    // 2️⃣ Decide colour based on numeric score
  const getScoreClass = (score: string | null) => {
    const value = getScoreValue(score);
    if (value === null) return "";

    if (value >= 75) return "score-green";
    if (value >= 40) return "score-yellow";
    return "score-red";
  };


  // 📄 PDF DOWNLOAD HANDLER
  const downloadPDF = async () => {
    if (!resumeId || !jdId) return;

    try {
      const blob = await generatePDF(resumeId, jdId);
      const url = window.URL.createObjectURL(blob);

      const a = document.createElement("a");
      a.href = url;
      a.download = "resume_match_report.pdf";
      a.click();

      window.URL.revokeObjectURL(url);
    } catch (err) {
      alert("Failed to download PDF report");
    }
  };

  const runMatch = async () => {
    if (!resumeId || !jdId) return;

    setLoading(true);
    try {
      const res = await getMatchAnalysis(resumeId, jdId);
      setResult(res);
    } catch (err) {
      console.error(err);
      setResult({ error: "Failed to load match analysis" });
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (resumeId && jdId) runMatch();
  }, [resumeId, jdId]);

  const highlightKeywords = (text: string, keywords: string[]) => {
    if (!text) return text;
    let highlighted = text;

    keywords.forEach((word) => {
      const pattern = new RegExp(`\\b${word}\\b`, "gi");
      highlighted = highlighted.replace(
        pattern,
        `<span class="highlight">${word}</span>`
      );
    });

    return highlighted;
  };

  const keywords = [
    "Java","Python","Spring","Spring Boot","REST","API","SQL","MySQL",
    "PostgreSQL","MongoDB","AWS","Docker","Kubernetes","Microservices",
    "React","Node","CI/CD"
  ];

  return (
    <div className="match-card">
      <h3 className="title">📊 Match Analysis</h3>

      {!resumeId || !jdId ? (
        <p className="hint">Upload Resume & JD to generate analysis</p>
      ) : (
        <button className="btn" onClick={runMatch} disabled={loading}>
          {loading ? <div className="spinner-inline"></div> : "Re-run Analysis"}
        </button>
      )}

      {/* 📄 Show PDF button ONLY when result exists */}
      {result && !result.error && (
        <button className="btn secondary" onClick={downloadPDF}>
          📄 Download PDF Report
        </button>
      )}

      {result && (
        <div className="result-panel fade-in">
          {result.error ? (
            <p className="error">{result.error}</p>
          ) : (
            <>
              {(() => {
  const scoreValue = getScoreValue(result.match_score);

              return (
               <div className={`score-badge ${getScoreClass(result.match_score)}`}>
                {scoreValue !== null ? `${scoreValue}%` : "--"}
                </div>
                 );
                 })()}


              {result.category_scores && (
                <div className="category-box">
                  <h4>📌 Category Scores</h4>
                  {Object.entries(result.category_scores).map(
                    ([name, value], index) => (
                      <div className="category-row" key={index}>
                        <span className="category-name">{name}</span>
                        <div className="progress-bar">
                          <div
                            className="progress-fill"
                            style={{ width: `${value}%` }}
                          />
                        </div>
                        <span className="category-value">{value}%</span>
                      </div>
                    )
                  )}
                </div>
              )}

              <div className="section strengths">
                <h4>✅ Strengths</h4>
                <ul>
                  {result.strengths?.length ? (
                    result.strengths.map((s: string, i: number) => (
                      <li
                        key={i}
                        dangerouslySetInnerHTML={{
                          __html: highlightKeywords(s, keywords),
                        }}
                      />
                    ))
                  ) : (
                    <p className="none">None found</p>
                  )}
                </ul>
              </div>

              <div className="section gaps">
                <h4>❌ Gaps</h4>
                <ul>
                  {result.gaps?.length ? (
                    result.gaps.map((g: string, i: number) => (
                      <li
                        key={i}
                        dangerouslySetInnerHTML={{
                          __html: highlightKeywords(g, keywords),
                        }}
                      />
                    ))
                  ) : (
                    <p className="none">None found</p>
                  )}
                </ul>
              </div>

              <div
                className="insights-box"
                dangerouslySetInnerHTML={{
                  __html: highlightKeywords(result.insights, keywords),
                }}
              />
            </>
          )}
        </div>
      )}
    </div>
  );
};

export default MatchScore;
