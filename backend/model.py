import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer
import joblib
import os

MODEL_PATH    = "ufc_model.joblib"
FIGHTERS_PATH = "fighters.joblib"

# ── All features including new scraped stats ──────────────────────────────────
FEATURE_COLS = [
    # Physical
    "age_diff",
    "reach_diff",
    "height_diff",
    "weight_diff",
    # Record
    "win_rate_diff",
    "win_streak_diff",
    "total_fights_diff",
    "finish_rate_diff",
    # Striking
    "sig_str_acc_diff",
    "sig_str_def_diff",
    "slpm_diff",          # strikes landed per min
    "sapm_diff",          # strikes absorbed per min
    # Grappling
    "td_acc_diff",
    "td_def_diff",
    "td_avg_diff",        # takedowns per 15 min
    "sub_avg_diff",
    # Finishing
    "ko_avg_diff",
    "ko_rate_diff",       # KO wins / total fights (from scraper)
    "sub_rate_diff",      # Sub wins / total fights (from scraper)
]


def load_and_clean(path: str = "data/ufc-master.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} fights, {df.shape[1]} columns")
    df = df[df["Winner"].isin(["Red", "Blue"])].copy()
    df["label"] = (df["Winner"] == "Red").astype(int)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    def safe_rate(wins, total):
        return np.where(total > 0, wins / total, 0.0)

    # Win rates
    df["R_win_rate"] = safe_rate(df["R_wins"], df["R_wins"] + df["R_losses"] + df.get("R_draws", 0))
    df["B_win_rate"] = safe_rate(df["B_wins"], df["B_wins"] + df["B_losses"] + df.get("B_draws", 0))

    # Finish rates
    df["R_finish_rate"] = safe_rate(
        df.get("R_win_by_KO/TKO", 0) + df.get("R_win_by_Submission", 0), df["R_wins"]
    )
    df["B_finish_rate"] = safe_rate(
        df.get("B_win_by_KO/TKO", 0) + df.get("B_win_by_Submission", 0), df["B_wins"]
    )

    # KO rate and Sub rate separately
    df["R_ko_rate"]  = safe_rate(df.get("R_win_by_KO/TKO",     0), df["R_wins"] + df["R_losses"])
    df["B_ko_rate"]  = safe_rate(df.get("B_win_by_KO/TKO",     0), df["B_wins"] + df["B_losses"])
    df["R_sub_rate"] = safe_rate(df.get("R_win_by_Submission",  0), df["R_wins"] + df["R_losses"])
    df["B_sub_rate"] = safe_rate(df.get("B_win_by_Submission",  0), df["B_wins"] + df["B_losses"])

    # Total fights
    df["R_total_fights"] = df["R_wins"] + df["R_losses"] + df.get("R_draws", pd.Series(0, index=df.index))
    df["B_total_fights"] = df["B_wins"] + df["B_losses"] + df.get("B_draws", pd.Series(0, index=df.index))

    # ── Differential features ──
    df["age_diff"]          = df["R_age"]                            - df["B_age"]
    df["reach_diff"]        = df["R_Reach_cms"]                      - df["B_Reach_cms"]
    df["height_diff"]       = df["R_Height_cms"]                     - df["B_Height_cms"]
    df["weight_diff"]       = df.get("R_Weight_lbs", 0)              - df.get("B_Weight_lbs", 0)
    df["win_rate_diff"]     = df["R_win_rate"]                       - df["B_win_rate"]
    df["win_streak_diff"]   = df.get("R_current_win_streak", 0)      - df.get("B_current_win_streak", 0)
    df["total_fights_diff"] = df["R_total_fights"]                   - df["B_total_fights"]
    df["finish_rate_diff"]  = df["R_finish_rate"]                    - df["B_finish_rate"]
    df["sig_str_acc_diff"]  = df.get("R_avg_SIG_STR_pct",      0)   - df.get("B_avg_SIG_STR_pct",      0)
    df["sig_str_def_diff"]  = df.get("R_avg_SIG_STR_att",      0)   - df.get("B_avg_SIG_STR_att",      0)
    df["slpm_diff"]         = df.get("R_avg_SIG_STR_landed",   0)   - df.get("B_avg_SIG_STR_landed",   0)
    df["sapm_diff"]         = df.get("R_avg_SIG_STR_att",      0)   - df.get("B_avg_SIG_STR_att",      0)
    df["td_acc_diff"]       = df.get("R_avg_TD_pct",           0)   - df.get("B_avg_TD_pct",           0)
    df["td_def_diff"]       = df.get("R_avg_TD_att",           0)   - df.get("B_avg_TD_att",           0)
    df["td_avg_diff"]       = df.get("R_avg_TD_landed",        0)   - df.get("B_avg_TD_landed",        0)
    df["sub_avg_diff"]      = df.get("R_avg_SUB_ATT",          0)   - df.get("B_avg_SUB_ATT",          0)
    df["ko_avg_diff"]       = df.get("R_avg_KD",               0)   - df.get("B_avg_KD",               0)
    df["ko_rate_diff"]      = df["R_ko_rate"]                        - df["B_ko_rate"]
    df["sub_rate_diff"]     = df["R_sub_rate"]                       - df["B_sub_rate"]

    return df


def build_fighter_profiles(df: pd.DataFrame) -> dict:
    """Fallback — only used if no scraped fighters.joblib exists."""
    fighters = {}

    if "date" in df.columns:
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values("date", ascending=True)

    for _, row in df.iterrows():
        for corner in ["R", "B"]:
            name = row.get(f"{corner}_fighter")
            if not isinstance(name, str) or not name.strip():
                continue

            existing = fighters.get(name, {})

            def pick(key, new_val):
                if new_val is not None and not (isinstance(new_val, float) and np.isnan(new_val)):
                    return new_val
                return existing.get(key)

            fighters[name] = {
                "name":         name,
                "age":          pick("age",         row.get(f"{corner}_age")),
                "height_cms":   pick("height_cms",  row.get(f"{corner}_Height_cms")),
                "reach_cms":    pick("reach_cms",   row.get(f"{corner}_Reach_cms")),
                "weight_lbs":   pick("weight_lbs",  row.get(f"{corner}_Weight_lbs")),
                "wins":         pick("wins",         row.get(f"{corner}_wins")),
                "losses":       pick("losses",       row.get(f"{corner}_losses")),
                "win_streak":   pick("win_streak",  row.get(f"{corner}_current_win_streak")),
                "sig_str_acc":  pick("sig_str_acc", row.get(f"{corner}_avg_SIG_STR_pct")),
                "td_acc":       pick("td_acc",       row.get(f"{corner}_avg_TD_pct")),
                "sub_avg":      pick("sub_avg",      row.get(f"{corner}_avg_SUB_ATT")),
                "ko_avg":       pick("ko_avg",       row.get(f"{corner}_avg_KD")),
                "stance":       pick("stance",       row.get(f"{corner}_Stance")),
                "weight_class": pick("weight_class", row.get("weight_class")),
            }

    return fighters


def train(data_path: str = "data/ufc-master.csv"):
    print("📦 Loading data...")
    df = load_and_clean(data_path)

    print("⚙️  Engineering features...")
    df = engineer_features(df)

    # ── Balance: flip every fight to remove red-corner bias ──
    df_flipped = df.copy()
    for col in FEATURE_COLS:
        df_flipped[col] = -df[col]
    df_flipped["label"] = 1 - df["label"]

    df_balanced = pd.concat([df, df_flipped], ignore_index=True)
    print(f"  Original: {len(df)} fights → Balanced: {len(df_balanced)} (2x flipped)")

    X = df_balanced[FEATURE_COLS]
    y = df_balanced["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Ensemble: Random Forest + Gradient Boosting voted together
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_split=10,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    gb = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        random_state=42,
    )
    ensemble = VotingClassifier(
        estimators=[("rf", rf), ("gb", gb)],
        voting="soft",
    )

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("model",   ensemble),
    ])

    print("🏋️  Training ensemble (Random Forest + Gradient Boosting)...")
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    acc   = accuracy_score(y_test, preds)
    print(f"\n✅ Test accuracy: {acc:.3f}")
    print(classification_report(y_test, preds, target_names=["Blue wins", "Red wins"]))

    joblib.dump(pipeline, MODEL_PATH)
    print(f"💾 Model saved → {MODEL_PATH}")

    # Keep scraped fighters if they exist
    if os.path.exists(FIGHTERS_PATH):
        fighters = joblib.load(FIGHTERS_PATH)
        print(f"ℹ️  Keeping existing fighter profiles → {FIGHTERS_PATH} ({len(fighters)} fighters)")
    else:
        fighters = build_fighter_profiles(df)
        joblib.dump(fighters, FIGHTERS_PATH)
        print(f"💾 Fighter profiles saved → {FIGHTERS_PATH} ({len(fighters)} fighters)")

    return pipeline, fighters


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model not found — run: python model.py")
    return joblib.load(MODEL_PATH)


def load_fighters():
    if not os.path.exists(FIGHTERS_PATH):
        raise FileNotFoundError("Fighters not found — run: python model.py")
    return joblib.load(FIGHTERS_PATH)


def predict_fight(fighter_red: dict, fighter_blue: dict, model) -> dict:
    def g(d, k, default=0.0):
        v = d.get(k)
        return float(v) if v is not None and not (isinstance(v, float) and np.isnan(v)) else default

    features = {
        "age_diff":          g(fighter_red, "age")          - g(fighter_blue, "age"),
        "reach_diff":        g(fighter_red, "reach_cms")     - g(fighter_blue, "reach_cms"),
        "height_diff":       g(fighter_red, "height_cms")    - g(fighter_blue, "height_cms"),
        "weight_diff":       g(fighter_red, "weight_lbs")    - g(fighter_blue, "weight_lbs"),
        "win_rate_diff":     _win_rate(fighter_red)          - _win_rate(fighter_blue),
        "win_streak_diff":   g(fighter_red, "win_streak")    - g(fighter_blue, "win_streak"),
        "total_fights_diff": _total(fighter_red)             - _total(fighter_blue),
        "finish_rate_diff":  g(fighter_red, "finish_rate")   - g(fighter_blue, "finish_rate"),
        "sig_str_acc_diff":  g(fighter_red, "sig_str_acc")   - g(fighter_blue, "sig_str_acc"),
        "sig_str_def_diff":  g(fighter_red, "sig_str_def")   - g(fighter_blue, "sig_str_def"),
        "slpm_diff":         g(fighter_red, "slpm")          - g(fighter_blue, "slpm"),
        "sapm_diff":         g(fighter_red, "sapm")          - g(fighter_blue, "sapm"),
        "td_acc_diff":       g(fighter_red, "td_acc")        - g(fighter_blue, "td_acc"),
        "td_def_diff":       g(fighter_red, "td_def")        - g(fighter_blue, "td_def"),
        "td_avg_diff":       g(fighter_red, "td_avg")        - g(fighter_blue, "td_avg"),
        "sub_avg_diff":      g(fighter_red, "sub_avg")       - g(fighter_blue, "sub_avg"),
        "ko_avg_diff":       g(fighter_red, "ko_avg")        - g(fighter_blue, "ko_avg"),
        "ko_rate_diff":      g(fighter_red, "ko_avg")        - g(fighter_blue, "ko_avg"),
        "sub_rate_diff":     g(fighter_red, "sub_avg")       - g(fighter_blue, "sub_avg"),
    }

    X         = pd.DataFrame([features])[FEATURE_COLS]
    proba     = model.predict_proba(X)[0]
    red_conf  = float(proba[1])
    blue_conf = float(proba[0])
    winner    = fighter_red["name"] if red_conf >= blue_conf else fighter_blue["name"]

    return {
        "winner":          winner,
        "red_confidence":  round(red_conf  * 100, 1),
        "blue_confidence": round(blue_conf * 100, 1),
        "features":        features,
    }


def _win_rate(f: dict) -> float:
    w     = f.get("wins")   or 0
    l     = f.get("losses") or 0
    total = w + l
    return w / total if total > 0 else 0.0


def _total(f: dict) -> float:
    return (f.get("wins") or 0) + (f.get("losses") or 0)


if __name__ == "__main__":
    train(r"data\ufc-master.csv\data.csv")