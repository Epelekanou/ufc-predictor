export default function FighterCard({ fighter, corner }) {
  const accent    = corner === "red" ? "#ff3d3d" : "#3d8bff";
  const avatarCls = corner === "red" ? "red-avatar" : "blue-avatar";
  const initials  = fighter.name.split(" ").map(w => w[0]).join("").slice(0, 2).toUpperCase();

  const wins   = fighter.wins   ?? null;
  const losses = fighter.losses ?? null;
  const record = (wins !== null && losses !== null) ? `${wins} - ${losses}` : "N/A";

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

  // Group stats into sections
  const physicalStats = [
    { label: "Height",    value: fmt(fighter.height_cms, " cm") },
    { label: "Reach",     value: fmt(fighter.reach_cms,  " cm") },
    { label: "Weight",    value: fmt(fighter.weight_lbs, " lbs") },
    { label: "Stance",    value: fighter.stance },
    { label: "Age",       value: fmt(fighter.age) },
    { label: "Streak",    value: fmt(fighter.win_streak) },
  ];

  const strikingStats = [
    { label: "Str. Acc.",  value: fmtPct(fighter.sig_str_acc) },
    { label: "Str. Def.",  value: fmtPct(fighter.sig_str_def) },
    { label: "SLpM",       value: fmtDec(fighter.slpm) },
    { label: "SApM",       value: fmtDec(fighter.sapm) },
  ];

  const grapplingStats = [
    { label: "TD Acc.",    value: fmtPct(fighter.td_acc) },
    { label: "TD Def.",    value: fmtPct(fighter.td_def) },
    { label: "TD Avg.",    value: fmtDec(fighter.td_avg) },
    { label: "Sub Avg.",   value: fmtDec(fighter.sub_avg) },
  ];

  const finishStats = [
    { label: "KO Avg.",      value: fmtDec(fighter.ko_avg) },
    { label: "Finish Rate",  value: fmtPct(fighter.finish_rate) },
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

      {/* Record badge */}
      <div className={`${corner}-corner`}>
        <div className="record-badge">{record}</div>
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
