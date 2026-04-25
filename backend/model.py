"""
UFC Fight Predictor — Upgraded Model v2
Improvements over v1:
  - XGBoost + LightGBM added to ensemble (stacking)
  - New features: recent form (last 3 fights), days since last fight, opponent quality
  - Time-based train/test split (more realistic than random)
  - Optuna hyperparameter tuning
  - Better calibration tracking

Run:  python model.py
"""

import pandas as pd
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

from datetime import datetime

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.impute import SimpleImputer

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("⚠ XGBoost not installed — run: pip install xgboost")

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("⚠ LightGBM not installed — run: pip install lightgbm")

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    print("⚠ Optuna not installed — run: pip install optuna")

MODEL_PATH    = "ufc_model.joblib"
FIGHTERS_PATH = "fighters.joblib"

# ── Feature columns (19 original + 6 new) ─────────────────────────────────────
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
    "slpm_diff",
    "sapm_diff",
    # Grappling
    "td_acc_diff",
    "td_def_diff",
    "td_avg_diff",
    "sub_avg_diff",
    # Finishing
    "ko_avg_diff",
    "ko_rate_diff",
    "sub_rate_diff",
    # ── NEW features ──────────────────────
    "recent_win_rate_diff",     # win rate in last 3 fights
    "days_since_fight_diff",    # who is more ring-rusty
    "experience_diff",          # total fights experience gap
    "title_fight",              # 1 if title fight, 0 otherwise
    "is_main_event",            # 1 if main event
    "avg_opponent_wins_diff",   # strength of schedule proxy
]


def load_and_clean(path: str = "data/ufc-master.csv/data.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"  Loaded {len(df)} fights, {df.shape[1]} columns")
    df = df[df["Winner"].isin(["Red", "Blue"])].copy()
    df["label"] = (df["Winner"] == "Red").astype(int)

    # Parse date for time-based split
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values("date", ascending=True).reset_index(drop=True)

    return df


def safe_rate(wins, total):
    return np.where(total > 0, wins / total, 0.0)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    # ── Original features ──────────────────────────────────────────────────────
    df["R_win_rate"] = safe_rate(df["R_wins"], df["R_wins"] + df["R_losses"] + df.get("R_draws", 0))
    df["B_win_rate"] = safe_rate(df["B_wins"], df["B_wins"] + df["B_losses"] + df.get("B_draws", 0))

    df["R_finish_rate"] = safe_rate(
        df.get("R_win_by_KO/TKO", 0) + df.get("R_win_by_Submission", 0), df["R_wins"]
    )
    df["B_finish_rate"] = safe_rate(
        df.get("B_win_by_KO/TKO", 0) + df.get("B_win_by_Submission", 0), df["B_wins"]
    )

    df["R_ko_rate"]  = safe_rate(df.get("R_win_by_KO/TKO",    0), df["R_wins"] + df["R_losses"])
    df["B_ko_rate"]  = safe_rate(df.get("B_win_by_KO/TKO",    0), df["B_wins"] + df["B_losses"])
    df["R_sub_rate"] = safe_rate(df.get("R_win_by_Submission", 0), df["R_wins"] + df["R_losses"])
    df["B_sub_rate"] = safe_rate(df.get("B_win_by_Submission", 0), df["B_wins"] + df["B_losses"])

    df["R_total_fights"] = df["R_wins"] + df["R_losses"] + df.get("R_draws", pd.Series(0, index=df.index))
    df["B_total_fights"] = df["B_wins"] + df["B_losses"] + df.get("B_draws", pd.Series(0, index=df.index))

    df["age_diff"]          = df["R_age"]                          - df["B_age"]
    df["reach_diff"]        = df["R_Reach_cms"]                    - df["B_Reach_cms"]
    df["height_diff"]       = df["R_Height_cms"]                   - df["B_Height_cms"]
    df["weight_diff"]       = df.get("R_Weight_lbs", 0)            - df.get("B_Weight_lbs", 0)
    df["win_rate_diff"]     = df["R_win_rate"]                     - df["B_win_rate"]
    df["win_streak_diff"]   = df.get("R_current_win_streak", 0)    - df.get("B_current_win_streak", 0)
    df["total_fights_diff"] = df["R_total_fights"]                 - df["B_total_fights"]
    df["finish_rate_diff"]  = df["R_finish_rate"]                  - df["B_finish_rate"]
    df["sig_str_acc_diff"]  = df.get("R_avg_SIG_STR_pct",    0)   - df.get("B_avg_SIG_STR_pct",    0)
    df["sig_str_def_diff"]  = df.get("R_avg_SIG_STR_att",    0)   - df.get("B_avg_SIG_STR_att",    0)
    df["slpm_diff"]         = df.get("R_avg_SIG_STR_landed", 0)   - df.get("B_avg_SIG_STR_landed", 0)
    df["sapm_diff"]         = df.get("R_avg_SIG_STR_att",    0)   - df.get("B_avg_SIG_STR_att",    0)
    df["td_acc_diff"]       = df.get("R_avg_TD_pct",         0)   - df.get("B_avg_TD_pct",         0)
    df["td_def_diff"]       = df.get("R_avg_TD_att",         0)   - df.get("B_avg_TD_att",         0)
    df["td_avg_diff"]       = df.get("R_avg_TD_landed",      0)   - df.get("B_avg_TD_landed",      0)
    df["sub_avg_diff"]      = df.get("R_avg_SUB_ATT",        0)   - df.get("B_avg_SUB_ATT",        0)
    df["ko_avg_diff"]       = df.get("R_avg_KD",             0)   - df.get("B_avg_KD",             0)
    df["ko_rate_diff"]      = df["R_ko_rate"]                      - df["B_ko_rate"]
    df["sub_rate_diff"]     = df["R_sub_rate"]                     - df["B_sub_rate"]

    # ── NEW Feature 1: Recent form (last 3 fights win rate) ───────────────────
    # Approximated from win streak and total fights
    df["R_recent_win_rate"] = np.where(
        df.get("R_current_win_streak", 0) >= 3, 1.0,
        np.where(df["R_total_fights"] > 0,
                 df.get("R_current_win_streak", 0) / df["R_total_fights"].clip(lower=1),
                 0.5)
    )
    df["B_recent_win_rate"] = np.where(
        df.get("B_current_win_streak", 0) >= 3, 1.0,
        np.where(df["B_total_fights"] > 0,
                 df.get("B_current_win_streak", 0) / df["B_total_fights"].clip(lower=1),
                 0.5)
    )
    df["recent_win_rate_diff"] = df["R_recent_win_rate"] - df["B_recent_win_rate"]

    # ── NEW Feature 2: Days since last fight (ring rust) ─────────────────────
    if "date" in df.columns and "R_days_since_last_fight" in df.columns:
        df["days_since_fight_diff"] = (
            df.get("R_days_since_last_fight", 0) - df.get("B_days_since_last_fight", 0)
        )
    else:
        df["days_since_fight_diff"] = 0.0

    # ── NEW Feature 3: Experience gap ─────────────────────────────────────────
    df["experience_diff"] = df["R_total_fights"] - df["B_total_fights"]

    # ── NEW Feature 4: Title fight flag ───────────────────────────────────────
    if "title_bout" in df.columns:
        df["title_fight"] = df["title_bout"].astype(int)
    elif "Fight_type" in df.columns:
        df["title_fight"] = df["Fight_type"].str.contains("Title", case=False, na=False).astype(int)
    else:
        df["title_fight"] = 0

    # ── NEW Feature 5: Main event flag ────────────────────────────────────────
    if "is_title_bout" in df.columns:
        df["is_main_event"] = df["is_title_bout"].astype(int)
    else:
        df["is_main_event"] = df["title_fight"]  # fallback

    # ── NEW Feature 6: Opponent quality (strength of schedule proxy) ──────────
    # Higher opponent wins = tougher schedule
    if "R_avg_opp_wins" in df.columns:
        df["avg_opponent_wins_diff"] = df.get("R_avg_opp_wins", 0) - df.get("B_avg_opp_wins", 0)
    else:
        # Approximate: fighters with more total fights faced tougher opponents on average
        df["avg_opponent_wins_diff"] = (df["R_total_fights"] * 0.5) - (df["B_total_fights"] * 0.5)

    return df


def time_based_split(df: pd.DataFrame, test_years: int = 2):
    """
    Split data by time instead of randomly.
    Train on older fights, test on recent ones.
    This is more realistic — you never know future fight results.
    """
    if "date" not in df.columns or df["date"].isna().all():
        print("  ⚠ No date column — falling back to random 80/20 split")
        from sklearn.model_selection import train_test_split
        return train_test_split(df, test_size=0.2, random_state=42)

    cutoff = df["date"].max() - pd.DateOffset(years=test_years)
    train  = df[df["date"] <= cutoff]
    test   = df[df["date"] >  cutoff]
    print(f"  Time split: {len(train)} train fights (≤{cutoff.date()}) | {len(test)} test fights (>{cutoff.date()})")
    return train, test


def tune_xgboost(X_train, y_train, n_trials: int = 30):
    """Use Optuna to find best XGBoost hyperparameters."""
    if not HAS_OPTUNA or not HAS_XGB:
        return {"n_estimators": 300, "max_depth": 5, "learning_rate": 0.05, "subsample": 0.8}

    print(f"  🔍 Tuning XGBoost with Optuna ({n_trials} trials)...")

    def objective(trial):
        params = {
            "n_estimators":       trial.suggest_int("n_estimators", 100, 500),
            "max_depth":          trial.suggest_int("max_depth", 3, 8),
            "learning_rate":      trial.suggest_float("learning_rate", 0.01, 0.2),
            "subsample":          trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":   trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight":   trial.suggest_int("min_child_weight", 1, 10),
            "random_state":       42,
            "eval_metric":        "logloss",
        }
        model  = xgb.XGBClassifier(**params)
        scores = cross_val_score(model, X_train, y_train, cv=3, scoring="accuracy", n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    print(f"  ✅ Best XGBoost accuracy (CV): {study.best_value:.3f}")
    return study.best_params


def build_pipeline(X_train, y_train):
    """Build stacking ensemble: RF + XGB + LGB → Logistic Regression meta."""

    # ── Base estimators ────────────────────────────────────────────────────────
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

    estimators = [("rf", rf), ("gb", gb)]

    # Add XGBoost if available
    if HAS_XGB:
        best_xgb_params = tune_xgboost(X_train, y_train, n_trials=30)
        best_xgb_params.update({"random_state": 42, "eval_metric": "logloss"})
        xgb_model = xgb.XGBClassifier(**best_xgb_params)
        estimators.append(("xgb", xgb_model))
        print("  ✅ XGBoost added to ensemble")

    # Add LightGBM if available
    if HAS_LGB:
        lgb_model = lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            verbose=-1,
        )
        estimators.append(("lgb", lgb_model))
        print("  ✅ LightGBM added to ensemble")

    # ── Stacking: base models → Logistic Regression meta-learner ──────────────
    stacking = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=1000, random_state=42),
        cv=5,
        stack_method="predict_proba",
        n_jobs=-1,
    )

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("model",   stacking),
    ])

    return pipeline


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


def train(data_path: str = "data/ufc-master.csv/data.csv"):
    print("📦 Loading data...")
    df = load_and_clean(data_path)

    print("⚙️  Engineering features...")
    df = engineer_features(df)

    # ── Balance: flip every fight to remove red-corner bias ───────────────────
    df_flipped = df.copy()
    for col in FEATURE_COLS:
        if col in df_flipped.columns:
            df_flipped[col] = -df[col]
    df_flipped["label"] = 1 - df["label"]

    df_balanced = pd.concat([df, df_flipped], ignore_index=True)
    print(f"  Original: {len(df)} fights → Balanced: {len(df_balanced)} (2x flipped)")

    # ── Time-based split ──────────────────────────────────────────────────────
    train_df, test_df = time_based_split(df_balanced, test_years=1)

    # Fill missing feature cols with 0
    for col in FEATURE_COLS:
        if col not in df_balanced.columns:
            train_df[col] = 0
            test_df[col]  = 0

    X_train = train_df[FEATURE_COLS]
    y_train = train_df["label"]
    X_test  = test_df[FEATURE_COLS]
    y_test  = test_df["label"]

    # ── Build & train stacking ensemble ──────────────────────────────────────
    print("\n🏋️  Building stacking ensemble...")
    pipeline = build_pipeline(X_train, y_train)

    print("🏋️  Training... (this may take 2-5 minutes)")
    pipeline.fit(X_train, y_train)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    preds = pipeline.predict(X_test)
    proba = pipeline.predict_proba(X_test)[:, 1]
    acc   = accuracy_score(y_test, preds)
    auc   = roc_auc_score(y_test, proba)

    print(f"\n✅ Test accuracy : {acc:.3f}  ({acc*100:.1f}%)")
    print(f"✅ ROC-AUC score : {auc:.3f}  (1.0 = perfect, 0.5 = random)")
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

    def win_rate(f):
        w, l = f.get("wins") or 0, f.get("losses") or 0
        return w / (w + l) if (w + l) > 0 else 0.0

    def total(f):
        return (f.get("wins") or 0) + (f.get("losses") or 0)

    features = {
        "age_diff":               g(fighter_red, "age")          - g(fighter_blue, "age"),
        "reach_diff":             g(fighter_red, "reach_cms")     - g(fighter_blue, "reach_cms"),
        "height_diff":            g(fighter_red, "height_cms")    - g(fighter_blue, "height_cms"),
        "weight_diff":            g(fighter_red, "weight_lbs")    - g(fighter_blue, "weight_lbs"),
        "win_rate_diff":          win_rate(fighter_red)           - win_rate(fighter_blue),
        "win_streak_diff":        g(fighter_red, "win_streak")    - g(fighter_blue, "win_streak"),
        "total_fights_diff":      total(fighter_red)              - total(fighter_blue),
        "finish_rate_diff":       g(fighter_red, "finish_rate")   - g(fighter_blue, "finish_rate"),
        "sig_str_acc_diff":       g(fighter_red, "sig_str_acc")   - g(fighter_blue, "sig_str_acc"),
        "sig_str_def_diff":       g(fighter_red, "sig_str_def")   - g(fighter_blue, "sig_str_def"),
        "slpm_diff":              g(fighter_red, "slpm")          - g(fighter_blue, "slpm"),
        "sapm_diff":              g(fighter_red, "sapm")          - g(fighter_blue, "sapm"),
        "td_acc_diff":            g(fighter_red, "td_acc")        - g(fighter_blue, "td_acc"),
        "td_def_diff":            g(fighter_red, "td_def")        - g(fighter_blue, "td_def"),
        "td_avg_diff":            g(fighter_red, "td_avg")        - g(fighter_blue, "td_avg"),
        "sub_avg_diff":           g(fighter_red, "sub_avg")       - g(fighter_blue, "sub_avg"),
        "ko_avg_diff":            g(fighter_red, "ko_avg")        - g(fighter_blue, "ko_avg"),
        "ko_rate_diff":           g(fighter_red, "ko_avg")        - g(fighter_blue, "ko_avg"),
        "sub_rate_diff":          g(fighter_red, "sub_avg")       - g(fighter_blue, "sub_avg"),
        # New features
        "recent_win_rate_diff":   g(fighter_red, "win_streak")    - g(fighter_blue, "win_streak"),
        "days_since_fight_diff":  0.0,   # not available at prediction time
        "experience_diff":        total(fighter_red)              - total(fighter_blue),
        "title_fight":            0.0,   # unknown at prediction time
        "is_main_event":          0.0,
        "avg_opponent_wins_diff": 0.0,
    }

    X     = pd.DataFrame([features])[FEATURE_COLS]
    proba = model.predict_proba(X)[0]
    red_conf  = float(proba[1])
    blue_conf = float(proba[0])
    winner = fighter_red["name"] if red_conf >= blue_conf else fighter_blue["name"]

    return {
        "winner":          winner,
        "red_confidence":  round(red_conf  * 100, 1),
        "blue_confidence": round(blue_conf * 100, 1),
        "features":        features,
    }


if __name__ == "__main__":
    train(r"data\ufc-master.csv\data.csv")