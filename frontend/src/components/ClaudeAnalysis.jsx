export default function ClaudeAnalysis({ analysis, loading }) {
  // Don't render anything if skipped or empty
  if (!loading && (!analysis || analysis === "SKIP")) return null;

  return (
    <section className="analysis-section">
      <div className="result-title">🤖 AI Fight Analysis</div>
      {loading ? (
        <div className="analysis-loading">
          <div className="spinner" />
          <span>Analyzing the matchup...</span>
        </div>
      ) : (
        <div className="analysis-text">
          {analysis?.split("\n").filter(Boolean).map((para, i) => (
            <p key={i}>{para}</p>
          ))}
        </div>
      )}
    </section>
  );
}
