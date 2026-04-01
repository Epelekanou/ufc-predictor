import { useState } from "react";
import FighterSelect from "./components/FighterSelect";
import FighterCard from "./components/FighterCard";
import PredictionResult from "./components/PredictionResult";
import ClaudeAnalysis from "./components/ClaudeAnalysis";
import axios from "axios";
import "./App.css";

const API = "http://localhost:8000";

export default function App() {
  const [redFighter,  setRedFighter]  = useState(null);
  const [blueFighter, setBlueFighter] = useState(null);
  const [prediction,  setPrediction]  = useState(null);
  const [analysis,    setAnalysis]    = useState(null);
  const [loading,     setLoading]     = useState(false);
  const [analyzing,   setAnalyzing]   = useState(false);
  const [error,       setError]       = useState(null);

  const handlePredict = async () => {
    if (!redFighter || !blueFighter) return;
    setLoading(true);
    setError(null);
    setPrediction(null);
    setAnalysis(null);

    // Step 1 — ML prediction
    try {
      const { data } = await axios.post(`${API}/predict`, {
        red_fighter:  redFighter.name,
        blue_fighter: blueFighter.name,
      });
      setPrediction(data);
    } catch (e) {
      setError("Prediction failed. Make sure the backend is running.");
      setLoading(false);
      return;
    }

    // Step 2 — Claude analysis (optional, hidden if API key not set)
    try {
      setAnalyzing(true);
      const { data: ai } = await axios.post(`${API}/analyze`, {
        red_fighter:  redFighter.name,
        blue_fighter: blueFighter.name,
      });
      setAnalysis(ai.analysis);
    } catch (e) {
      setAnalysis("SKIP"); // hide section silently
    } finally {
      setLoading(false);
      setAnalyzing(false);
    }
  };

  const handleReset = () => {
    setRedFighter(null);
    setBlueFighter(null);
    setPrediction(null);
    setAnalysis(null);
    setError(null);
  };

  const showAnalysis = analyzing || (analysis && analysis !== "SKIP");

  return (
    <div className="app">
      <header className="header">
        <div className="header-inner">
          <span className="header-icon">🥊</span>
          <h1>UFC Fight Predictor</h1>
          <p className="header-sub">
            ML model trained on 6,000+ real UFC fights
          </p>
        </div>
      </header>

      <main className="main">
        {/* Fighter Selection */}
        <section className="selection-section">
          <div className="vs-grid">
            {/* Red Corner */}
            <div className="corner red-corner">
              <div className="corner-label">🔴 Red Corner</div>
              <FighterSelect corner="red" selected={redFighter} onSelect={setRedFighter} api={API} />
              {redFighter && <FighterCard fighter={redFighter} corner="red" />}
            </div>

            {/* VS */}
            <div className="vs-center">
              <div className="vs-badge">VS</div>
              {redFighter && blueFighter && (
                <button className="predict-btn" onClick={handlePredict} disabled={loading}>
                  {loading ? "..." : "⚡ Predict"}
                </button>
              )}
              {(redFighter || blueFighter) && (
                <button className="reset-btn" onClick={handleReset}>Reset</button>
              )}
            </div>

            {/* Blue Corner */}
            <div className="corner blue-corner">
              <div className="corner-label">🔵 Blue Corner</div>
              <FighterSelect corner="blue" selected={blueFighter} onSelect={setBlueFighter} api={API} />
              {blueFighter && <FighterCard fighter={blueFighter} corner="blue" />}
            </div>
          </div>
        </section>

        {error && <div className="error-box">{error}</div>}

        {prediction && <PredictionResult prediction={prediction} />}

        {showAnalysis && (
          <ClaudeAnalysis analysis={analysis} loading={analyzing} />
        )}
      </main>

      <footer className="footer">
        Built with FastAPI · scikit-learn · React
      </footer>
    </div>
  );
}
