// ── Style classifier (mirrors model.py logic) ─────────────────────────────────
function classifyStyle(fighter) {
  const slpm    = fighter.slpm    || 0;
  const sapm    = fighter.sapm    || 0;
  const tdAvg   = fighter.td_avg  || 0;
  const subAvg  = fighter.sub_avg || 0;
  const strAcc  = fighter.sig_str_acc || 0;

  const strikingScore  = (slpm * 0.4) + (strAcc * 0.3) + (Math.max(0, slpm - sapm) * 0.3);
  const grapplingScore = (tdAvg * 0.5) + (subAvg * 0.5);

  if (strikingScore > 3.0 && strikingScore > grapplingScore * 2) {
    return { label: "Striker",   icon: "🥊", color: "#ff3d3d", bg: "rgba(255,61,61,0.12)" };
  } else if (grapplingScore > 1.5 && grapplingScore > strikingScore * 0.5) {
    if (subAvg > tdAvg) {
      return { label: "Grappler", icon: "🔒", color: "#a78bfa", bg: "rgba(167,139,250,0.12)" };
    }
    return { label: "Wrestler",  icon: "💪", color: "#34d399", bg: "rgba(52,211,153,0.12)" };
  }
  return { label: "Balanced",  icon: "⚖️", color: "#ffc84a", bg: "rgba(255,200,74,0.12)" };
}

export default function FighterCard({ fighter, corner }) {
  const accent    = corner === "red" ? "#ff3d3d" : "#3d8bff";
  const avatarCls = corner === "red" ? "red-avatar" : "blue-avatar";
  const initials  = fighter.name.split(" ").map(w => w[0]).join("").slice(0, 2).toUpperCase();

  const wins   = fighter.wins   ?? null;
  const losses = fighter.losses ?? null;
  const record = (wins !== null && losses !== null) ? `${wins} - ${losses}` : "N/A";

  const style = classifyStyle(fighter);

  const fmt = (val, suffix = "") => {
    if (val === null || val === undefined) return null;
    return `${typeof val === "number" ? val : val}${suffix}`;
  };

  const fmtPct = (val) => {
    if (val === null || val === undefined) return null;
    const v = val > 1 ? val : val * 100;
    return `${Math.round(v)}%`;
  };

  const fmtDec = (val, decimals = 2) => {
    if (val === null || val === undefined) return null;
    return Number(val).toFixed(decimals);
  };

  const physicalStats = [
    { label: "Height",  value: fmt(fighter.height_cms, " cm") },
    { label: "Reach",   value: fmt(fighter.reach_cms,  " cm") },
    { label: "Weight",  value: fmt(fighter.weight_lbs, " lbs") },
    { label: "Stance",  value: fighter.stance },
    { label: "Age",     value: fmt(fighter.age) },
    { label: "Streak",  value: fmt(fighter.win_streak) },
  ];

  const strikingStats = [
    { label: "Str. Acc.", value: fmtPct(fighter.sig_str_acc) },
    { label: "Str. Def.", value: fmtPct(fighter.sig_str_def) },
    { label: "SLpM",      value: fmtDec(fighter.slpm) },
    { label: "SApM",      value: fmtDec(fighter.sapm) },
  ];

  const grapplingStats = [
    { label: "TD Acc.",  value: fmtPct(fighter.td_acc) },
    { label: "TD Def.",  value: fmtPct(fighter.td_def) },
    { label: "TD Avg.",  value: fmtDec(fighter.td_avg) },
    { label: "Sub Avg.", value: fmtDec(fighter.sub_avg) },
  ];

  const finishStats = [
    { label: "KO Avg.",     value: fmtDec(fighter.ko_avg) },
    { label: "Finish Rate", value: fmtPct(fighter.finish_rate) },
  ];

  return (
    <div className="fighter-card" style={{ borderColor: accent }}>

      {/* Header */}
      <div className="fighter-header">
        <div className={`fighter-avatar ${avatarCls}`}>{initials}</div>
        <div>
          <div className="fighter-name" style={{ color: accent }}>{fighter.name}</div>
          {fighter.weight_class && (
            <div className="weight-class">{fighter.weight_class}</div>
          )}
        </div>
      </div>

      {/* Record + Style badges row */}
      <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", marginBottom: "0.85rem", flexWrap: "wrap" }}>
        {/* Record */}
        <div className={`${corner}-corner`}>
          <div className="record-badge" style={{ marginBottom: 0 }}>{record}</div>
        </div>

        {/* Style badge */}
        <div style={{
          display:        "inline-flex",
          alignItems:     "center",
          gap:            "0.3rem",
          background:     style.bg,
          color:          style.color,
          border:         `1px solid ${style.color}40`,
          borderRadius:   "6px",
          padding:        "0.15rem 0.55rem",
          fontSize:       "0.72rem",
          fontWeight:     700,
          letterSpacing:  "0.06em",
          textTransform:  "uppercase",
        }}>
          <span>{style.icon}</span>
          <span>{style.label}</span>
        </div>
      </div>

      {/* Physical */}
      <div className="stats-section-label">Physical</div>
      <div className="stats-grid">
        {physicalStats.map(({ label, value }) => (
          <StatItem key={label} label={label} value={value} />
        ))}
      </div>

      {/* Striking */}
      <div className="stats-section-label" style={{ marginTop: "0.6rem" }}>Striking</div>
      <div className="stats-grid">
        {strikingStats.map(({ label, value }) => (
          <StatItem key={label} label={label} value={value} />
        ))}
      </div>

      {/* Grappling */}
      <div className="stats-section-label" style={{ marginTop: "0.6rem" }}>Grappling</div>
      <div className="stats-grid">
        {grapplingStats.map(({ label, value }) => (
          <StatItem key={label} label={label} value={value} />
        ))}
      </div>

      {/* Finishing */}
      <div className="stats-section-label" style={{ marginTop: "0.6rem" }}>Finishing</div>
      <div className="stats-grid">
        {finishStats.map(({ label, value }) => (
          <StatItem key={label} label={label} value={value} />
        ))}
      </div>
    </div>
  );
}

function StatItem({ label, value }) {
  return (
    <div className="stat-item">
      <span className="stat-label">{label}</span>
      <span className={`stat-value${!value ? " na" : ""}`}>
        {value ?? "—"}
      </span>
    </div>
  );
}
