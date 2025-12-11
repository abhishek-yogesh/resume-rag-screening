import React, { useState, useEffect } from "react";
import { getMatchAnalysis } from "../api/api";
import "./MatchScore.css";

interface MatchScoreProps {
  resumeId: string | null;
  jdId: string | null;
}

const MatchScore: React.FC<MatchScoreProps> = ({ resumeId, jdId }) => {
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  const getScoreClass = (score: string) => {
    const value = parseFloat(score);
    if (value >= 75) return "score-green";
    if (value >= 40) return "score-yellow";
    return "score-red";
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

      {result && (
        <div className="result-panel fade-in">
          {result.error ? (
            <p className="error">{result.error}</p>
          ) : (
            <>
              {/* MAIN SCORE BADGE */}
              <div className={`score-badge ${getScoreClass(result.match_score)}`}>
                {result.match_score}
              </div>

              {/* CATEGORY SCORES */}
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
                          ></div>
                        </div>

                        <span className="category-value">{value}%</span>
                      </div>
                    )
                  )}
                </div>
              )}

              {/* STRENGTHS */}
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

              {/* GAPS */}
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

              {/* INSIGHTS */}
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
