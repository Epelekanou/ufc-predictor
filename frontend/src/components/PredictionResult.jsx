import {
  RadarChart, Radar, PolarGrid, PolarAngleAxis,
  ResponsiveContainer, Tooltip,
  BarChart, Bar, XAxis, YAxis,CartesianGrid,
} from "recharts";

export default function PredictionResult({ prediction }) {
  const { red_fighter, blue_fighter, winner, red_confidence, blue_confidence } = prediction;
  const redWins = winner === red_fighter.name;

  // Radar — 8 dimensions covering all skill areas
  const radarData = [
    {
      stat: "Striking",
      red:  pct(red_fighter.sig_str_acc),
      blue: pct(blue_fighter.sig_str_acc),
    },
    {
      stat: "Defense",
      red:  pct(red_fighter.sig_str_def),
      blue: pct(blue_fighter.sig_str_def),
    },
    {
      stat: "Output",
      red:  norm(red_fighter.slpm,  10),
      blue: norm(blue_fighter.slpm, 10),
    },
    {
      stat: "TD Acc.",
      red:  pct(red_fighter.td_acc),
      blue: pct(blue_fighter.td_acc),
    },
    {
      stat: "TD Def.",
      red:  pct(red_fighter.td_def),
      blue: pct(blue_fighter.td_def),
    },
    {
      stat: "Win Rate",
      red:  winRate(red_fighter),
      blue: winRate(blue_fighter),
    },
    {
      stat: "Finishing",
      red:  pct(red_fighter.finish_rate),
      blue: pct(blue_fighter.finish_rate),
    },
    {
      stat: "Streak",
      red:  norm(red_fighter.win_streak,  10),
      blue: norm(blue_fighter.win_streak, 10),
    },
  ];

  // Bar chart — head to head stat comparison
  const barData = [
    {
      stat: "Str. Acc %",
      red:  pct(red_fighter.sig_str_acc),
      blue: pct(blue_fighter.sig_str_acc),
    },
    {
      stat: "Str. Def %",
      red:  pct(red_fighter.sig_str_def),
      blue: pct(blue_fighter.sig_str_def),
    },
    {
      stat: "TD Acc %",
      red:  pct(red_fighter.td_acc),
      blue: pct(blue_fighter.td_acc),
    },
    {
      stat: "TD Def %",
      red:  pct(red_fighter.td_def),
      blue: pct(blue_fighter.td_def),
    },
    {
      stat: "Finish %",
      red:  pct(red_fighter.finish_rate),
      blue: pct(blue_fighter.finish_rate),
    },
  ];

  return (
    <section className="result-section">
      <div className="result-title">⚡ Prediction</div>

      {/* Winner banner */}
      <div className={`winner-banner ${redWins ? "red-win" : "blue-win"}`}>
        <span className="winner-label">Predicted Winner</span>
        <span className="winner-name">{winner}</span>
      </div>

      {/* Confidence bar */}
      <div className="confidence-wrap">
        <div className="conf-labels">
          <span className="red-text">
            <span className="conf-name">{red_fighter.name}</span>
            <span className="conf-pct"> {red_confidence}%</span>
          </span>
          <span className="blue-text">
            <span className="conf-pct">{blue_confidence}% </span>
            <span className="conf-name">{blue_fighter.name}</span>
          </span>
        </div>
        <div className="conf-bar-track">
          <div className="conf-bar-red"  style={{ width: `${red_confidence}%` }} />
          <div className="conf-bar-blue" style={{ width: `${blue_confidence}%` }} />
        </div>
        <div className="conf-center-label">Win Probability</div>
      </div>

      {/* Two charts side by side */}
      <div className="charts-grid">
        {/* Radar */}
        <div>
          <div className="radar-title">Overall Profile (8 dimensions)</div>
          <ResponsiveContainer width="100%" height={280}>
            <RadarChart data={radarData} margin={{ top: 10, right: 30, bottom: 10, left: 30 }}>
              <PolarGrid stroke="#252d42" />
              <PolarAngleAxis
                dataKey="stat"
                tick={{ fill: "#8494b4", fontSize: 10, fontFamily: "Inter" }}
              />
              <Tooltip
                contentStyle={{ background: "#0f1520", border: "1px solid #252d42", borderRadius: 8, fontSize: 11 }}
                labelStyle={{ color: "#e8edf5", marginBottom: 4 }}
              />
              <Radar name={red_fighter.name}  dataKey="red"  stroke="#ff3d3d" fill="#ff3d3d" fillOpacity={0.2} strokeWidth={2} />
              <Radar name={blue_fighter.name} dataKey="blue" stroke="#3d8bff" fill="#3d8bff" fillOpacity={0.2} strokeWidth={2} />
            </RadarChart>
          </ResponsiveContainer>
          <div className="radar-legend">
            <span><span className="dot red-dot" />{red_fighter.name}</span>
            <span><span className="dot blue-dot" />{blue_fighter.name}</span>
          </div>
        </div>

        {/* Bar chart */}
        <div>
          <div className="radar-title">Head-to-Head Stats</div>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart
              data={barData}
              layout="vertical"
              margin={{ top: 5, right: 20, bottom: 5, left: 55 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#252d42" horizontal={false} />
              <XAxis type="number" domain={[0, 100]} tick={{ fill: "#5a6a8a", fontSize: 10 }} tickFormatter={v => `${v}%`} />
              <YAxis type="category" dataKey="stat" tick={{ fill: "#8494b4", fontSize: 10 }} />
              <Tooltip
                contentStyle={{ background: "#0f1520", border: "1px solid #252d42", borderRadius: 8, fontSize: 11 }}
                formatter={(v, name) => [`${v}%`, name]}
              />
              <Bar dataKey="red"  name={red_fighter.name}  fill="#ff3d3d" opacity={0.85} radius={[0,4,4,0]} />
              <Bar dataKey="blue" name={blue_fighter.name} fill="#3d8bff" opacity={0.85} radius={[0,4,4,0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Key advantages */}
      <div className="advantages-grid">
        <AdvantageCard
          fighter={red_fighter}
          opponent={blue_fighter}
          color="#ff3d3d"
          label="Red Corner Edge"
        />
        <AdvantageCard
          fighter={blue_fighter}
          opponent={red_fighter}
          color="#3d8bff"
          label="Blue Corner Edge"
        />
      </div>
    </section>
  );
}

function AdvantageCard({ fighter, opponent, color, label }) {
  const advantages = [];

  if ((fighter.reach_cms || 0) > (opponent.reach_cms || 0))
    advantages.push(`+${((fighter.reach_cms||0) - (opponent.reach_cms||0)).toFixed(1)}cm reach`);
  if (pct(fighter.sig_str_acc) > pct(opponent.sig_str_acc))
    advantages.push(`Better strike accuracy (${pct(fighter.sig_str_acc)}%)`);
  if (pct(fighter.sig_str_def) > pct(opponent.sig_str_def))
    advantages.push(`Better strike defense (${pct(fighter.sig_str_def)}%)`);
  if (pct(fighter.td_acc) > pct(opponent.td_acc))
    advantages.push(`Better TD accuracy (${pct(fighter.td_acc)}%)`);
  if (pct(fighter.td_def) > pct(opponent.td_def))
    advantages.push(`Better TD defense (${pct(fighter.td_def)}%)`);
  if ((fighter.win_streak || 0) > (opponent.win_streak || 0))
    advantages.push(`Longer win streak (${fighter.win_streak})`);
  if (pct(fighter.finish_rate) > pct(opponent.finish_rate))
    advantages.push(`Higher finish rate (${pct(fighter.finish_rate)}%)`);
  if (winRate(fighter) > winRate(opponent))
    advantages.push(`Better win rate (${winRate(fighter)}%)`);

  return (
    <div className="advantage-card" style={{ borderColor: color }}>
      <div className="advantage-label" style={{ color }}>{label}</div>
      <div className="advantage-name">{fighter.name}</div>
      {advantages.length > 0 ? (
        <ul className="advantage-list">
          {advantages.slice(0, 4).map((a, i) => (
            <li key={i} style={{ color: "#8494b4" }}>✓ {a}</li>
          ))}
        </ul>
      ) : (
        <p style={{ color: "#5a6a8a", fontSize: "0.8rem" }}>No clear stat advantages</p>
      )}
    </div>
  );
}

// ── Helpers ──
function pct(v) {
  if (v === null || v === undefined) return 0;
  return Math.round(v > 1 ? v : v * 100);
}
function winRate(f) {
  const t = (f.wins || 0) + (f.losses || 0);
  return t ? Math.round((f.wins / t) * 100) : 0;
}
function norm(v, max) {
  if (v === null || v === undefined) return 0;
  return Math.min(100, Math.round((v / max) * 100));
}
