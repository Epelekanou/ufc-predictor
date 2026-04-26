"""
UFC Fight Predictor — Model v3
New features added:
  1. Fighter style classification (striker/grappler/wrestler/all-rounder)
  2. Recent form — last 3 fights approximation
  3. Performance trend (recent vs career average)
  4. Strike differential (landed minus absorbed per min)
  5. Inactivity penalty (days since last fight)
  6. Weight class change indicator
  7. Home crowd advantage (fighter country vs event location)

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

# ── Feature columns ────────────────────────────────────────────────────────────
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
    # Previously added
    "recent_win_rate_diff",
    "days_since_fight_diff",
    "experience_diff",
    "title_fight",
    "is_main_event",
    "avg_opponent_wins_diff",
    # ── NEW 7 features ─────────────────────────────────────────────────────────
    "style_matchup",            # 1. striker vs grappler matchup score
    "R_is_striker",             # 1. red is striker
    "B_is_striker",             # 1. blue is striker
    "R_is_grappler",            # 1. red is grappler
    "B_is_grappler",            # 1. blue is grappler
    "recent_form_diff",         # 2. last 3 fights win rate diff
    "trend_str_diff",           # 3. recent striking trend vs career
    "trend_td_diff",            # 3. recent grappling trend vs career
    "str_differential_diff",    # 4. (SLpM - SApM) differential
    "inactivity_diff",          # 5. days since last fight diff
    "weight_class_change_R",    # 6. red fighter moved weight classes
    "weight_class_change_B",    # 6. blue fighter moved weight classes
    "home_advantage",           # 7. home crowd advantage score
]


def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"  Loaded {len(df)} fights, {df.shape[1]} columns")
    df = df[df["Winner"].isin(["Red", "Blue"])].copy()
    df["label"] = (df["Winner"] == "Red").astype(int)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values("date", ascending=True).reset_index(drop=True)

    return df


def safe_rate(wins, total):
    return np.where(total > 0, wins / total, 0.0)


def classify_style(slpm, sapm, td_avg, sub_avg, sig_str_acc):
    """
    Classify fighter style based on their stats.
    Returns: 0=balanced, 1=striker, 2=grappler, 3=wrestler
    """
    slpm      = slpm      or 0
    sapm      = sapm      or 0
    td_avg    = td_avg    or 0
    sub_avg   = sub_avg   or 0
    str_acc   = sig_str_acc or 0

    striking_score  = (slpm * 0.4) + (str_acc * 0.3) + (max(0, slpm - sapm) * 0.3)
    grappling_score = (td_avg * 0.5) + (sub_avg * 0.5)

    if striking_score > 3.0 and striking_score > grappling_score * 2:
        return 1  # striker
    elif grappling_score > 1.5 and grappling_score > striking_score * 0.5:
        if sub_avg > td_avg:
            return 2  # submission grappler
        return 3  # wrestler
    return 0  # balanced


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

    # Previous new features
    df["R_recent_win_rate"] = np.where(
        df.get("R_current_win_streak", 0) >= 3, 1.0,
        np.where(df["R_total_fights"] > 0,
                 df.get("R_current_win_streak", 0) / df["R_total_fights"].clip(lower=1), 0.5)
    )
    df["B_recent_win_rate"] = np.where(
        df.get("B_current_win_streak", 0) >= 3, 1.0,
        np.where(df["B_total_fights"] > 0,
                 df.get("B_current_win_streak", 0) / df["B_total_fights"].clip(lower=1), 0.5)
    )
    df["recent_win_rate_diff"] = df["R_recent_win_rate"] - df["B_recent_win_rate"]
    df["days_since_fight_diff"] = 0.0
    df["experience_diff"]  = df["R_total_fights"] - df["B_total_fights"]

    if "title_bout" in df.columns:
        df["title_fight"] = df["title_bout"].astype(int)
    else:
        df["title_fight"] = 0
    df["is_main_event"] = df["title_fight"]

    if "R_avg_opp_wins" in df.columns:
        df["avg_opponent_wins_diff"] = df.get("R_avg_opp_wins", 0) - df.get("B_avg_opp_wins", 0)
    else:
        df["avg_opponent_wins_diff"] = (df["R_total_fights"] * 0.5) - (df["B_total_fights"] * 0.5)

    # ── NEW Feature 1: Fighter Style Classification ────────────────────────────
    r_slpm    = df.get("R_avg_SIG_STR_landed", pd.Series(0, index=df.index))
    r_sapm    = df.get("R_avg_SIG_STR_att",    pd.Series(0, index=df.index))
    r_td      = df.get("R_avg_TD_landed",       pd.Series(0, index=df.index))
    r_sub     = df.get("R_avg_SUB_ATT",         pd.Series(0, index=df.index))
    r_stracc  = df.get("R_avg_SIG_STR_pct",     pd.Series(0, index=df.index))

    b_slpm    = df.get("B_avg_SIG_STR_landed", pd.Series(0, index=df.index))
    b_sapm    = df.get("B_avg_SIG_STR_att",    pd.Series(0, index=df.index))
    b_td      = df.get("B_avg_TD_landed",       pd.Series(0, index=df.index))
    b_sub     = df.get("B_avg_SUB_ATT",         pd.Series(0, index=df.index))
    b_stracc  = df.get("B_avg_SIG_STR_pct",     pd.Series(0, index=df.index))

    df["R_style"] = np.vectorize(classify_style)(r_slpm, r_sapm, r_td, r_sub, r_stracc)
    df["B_style"] = np.vectorize(classify_style)(b_slpm, b_sapm, b_td, b_sub, b_stracc)

    df["R_is_striker"]  = (df["R_style"] == 1).astype(int)
    df["B_is_striker"]  = (df["B_style"] == 1).astype(int)
    df["R_is_grappler"] = (df["R_style"].isin([2, 3])).astype(int)
    df["B_is_grappler"] = (df["B_style"].isin([2, 3])).astype(int)

    # Style matchup: striker vs grappler = 1 (grappler advantage historically)
    # grappler vs striker = -1, same style = 0
    df["style_matchup"] = np.where(
        (df["R_is_striker"] == 1) & (df["B_is_grappler"] == 1), -1,  # red striker vs blue grappler → blue favored
        np.where(
            (df["R_is_grappler"] == 1) & (df["B_is_striker"] == 1), 1,  # red grappler vs blue striker → red favored
            0  # same style
        )
    )

    # ── NEW Feature 2: Recent Form (last 3 fights better approximation) ────────
    # Use win streak relative to recent loss streak for better signal
    r_win_str  = df.get("R_current_win_streak",  pd.Series(0, index=df.index))
    r_lose_str = df.get("R_current_lose_streak", pd.Series(0, index=df.index))
    b_win_str  = df.get("B_current_win_streak",  pd.Series(0, index=df.index))
    b_lose_str = df.get("B_current_lose_streak", pd.Series(0, index=df.index))

    df["R_form_score"] = r_win_str - (r_lose_str * 1.5)  # losses weighted more
    df["B_form_score"] = b_win_str - (b_lose_str * 1.5)
    df["recent_form_diff"] = df["R_form_score"] - df["B_form_score"]

    # ── NEW Feature 3: Performance Trend (recent vs career) ───────────────────
    # If recent SIG STR acc > career average → improving
    # Approximate: win streak / total fights vs overall win rate
    df["R_trend"] = np.where(
        df["R_total_fights"] > 5,
        (r_win_str / df["R_total_fights"].clip(lower=1)) - df["R_win_rate"],
        0.0
    )
    df["B_trend"] = np.where(
        df["B_total_fights"] > 5,
        (b_win_str / df["B_total_fights"].clip(lower=1)) - df["B_win_rate"],
        0.0
    )
    df["trend_str_diff"] = df["R_trend"] - df["B_trend"]
    df["trend_td_diff"]  = df["R_trend"] - df["B_trend"]  # proxy

    # ── NEW Feature 4: Strike Differential (SLpM - SApM) ─────────────────────
    # Positive = landing more than absorbing = winning exchanges
    df["R_str_diff"] = r_slpm - r_sapm
    df["B_str_diff"] = b_slpm - b_sapm
    df["str_differential_diff"] = df["R_str_diff"] - df["B_str_diff"]

    # ── NEW Feature 5: Inactivity Penalty ─────────────────────────────────────
    if "R_days_since_last_fight" in df.columns:
        df["inactivity_diff"] = df["R_days_since_last_fight"] - df["B_days_since_last_fight"]
    else:
        df["inactivity_diff"] = 0.0

    # ── NEW Feature 6: Weight Class Change ────────────────────────────────────
    # Detect if fighter moved weight classes (approximated by weight vs typical class weight)
    WEIGHT_CLASS_LBS = {
        "Strawweight": 115, "Flyweight": 125, "Bantamweight": 135,
        "Featherweight": 145, "Lightweight": 155, "Welterweight": 170,
        "Middleweight": 185, "Light Heavyweight": 205, "Heavyweight": 265,
    }
    if "weight_class" in df.columns and "R_Weight_lbs" in df.columns:
        def weight_change(row, corner):
            wc     = str(row.get("weight_class", ""))
            wt     = row.get(f"{corner}_Weight_lbs", 0) or 0
            target = WEIGHT_CLASS_LBS.get(wc, wt)
            return 1 if abs(wt - target) > 15 else 0

        df["weight_class_change_R"] = df.apply(lambda r: weight_change(r, "R"), axis=1)
        df["weight_class_change_B"] = df.apply(lambda r: weight_change(r, "B"), axis=1)
    else:
        df["weight_class_change_R"] = 0
        df["weight_class_change_B"] = 0

    # ── NEW Feature 7: Home Crowd Advantage ───────────────────────────────────
    # Check if event location matches fighter's likely home country
    COUNTRY_KEYWORDS = {
        "USA": ["USA", "United States", "Las Vegas", "New York", "Los Angeles", "Houston", "Denver"],
        "Brazil": ["Brazil", "São Paulo", "Rio", "Fortaleza"],
        "UK": ["United Kingdom", "London", "Manchester", "Glasgow"],
        "Canada": ["Canada", "Toronto", "Vancouver", "Montreal"],
        "Australia": ["Australia", "Sydney", "Melbourne", "Perth"],
        "Russia": ["Russia", "Moscow"],
        "Ireland": ["Ireland", "Dublin"],
    }

    def home_advantage(row):
        location = str(row.get("location", ""))
        r_stance = str(row.get("R_Stance", ""))
        b_stance = str(row.get("B_Stance", ""))
        # Simple heuristic: if location contains a known country keyword
        # This is a rough proxy — real implementation needs fighter nationality data
        for country, keywords in COUNTRY_KEYWORDS.items():
            if any(kw.lower() in location.lower() for kw in keywords):
                return 0  # neutral — can't determine without fighter nationality
        return 0

    df["home_advantage"] = df.apply(home_advantage, axis=1)

    return df


def time_based_split(df: pd.DataFrame, test_years: int = 1):
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
    if not HAS_OPTUNA or not HAS_XGB:
        return {"n_estimators": 300, "max_depth": 5, "learning_rate": 0.05, "subsample": 0.8}

    print(f"  🔍 Tuning XGBoost with Optuna ({n_trials} trials)...")

    def objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 100, 500),
            "max_depth":        trial.suggest_int("max_depth", 3, 8),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.2),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "random_state":     42,
            "eval_metric":      "logloss",
        }
        model  = xgb.XGBClassifier(**params)
        scores = cross_val_score(model, X_train, y_train, cv=3, scoring="accuracy", n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    print(f"  ✅ Best XGBoost accuracy (CV): {study.best_value:.3f}")
    return study.best_params


def build_pipeline(X_train, y_train):
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_split=10,
        random_state=42, n_jobs=-1, class_weight="balanced",
    )
    gb = GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42,
    )
    estimators = [("rf", rf), ("gb", gb)]

    if HAS_XGB:
        best_xgb_params = tune_xgboost(X_train, y_train, n_trials=30)
        best_xgb_params.update({"random_state": 42, "eval_metric": "logloss"})
        estimators.append(("xgb", xgb.XGBClassifier(**best_xgb_params)))
        print("  ✅ XGBoost added to ensemble")

    if HAS_LGB:
        estimators.append(("lgb", lgb.LGBMClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            num_leaves=31, random_state=42, verbose=-1,
        )))
        print("  ✅ LightGBM added to ensemble")

    stacking = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=1000, random_state=42),
        cv=5, stack_method="predict_proba", n_jobs=-1,
    )

    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("model",   stacking),
    ])


def build_fighter_profiles(df: pd.DataFrame) -> dict:
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
                "age":          pick("age",        row.get(f"{corner}_age")),
                "height_cms":   pick("height_cms", row.get(f"{corner}_Height_cms")),
                "reach_cms":    pick("reach_cms",  row.get(f"{corner}_Reach_cms")),
                "weight_lbs":   pick("weight_lbs", row.get(f"{corner}_Weight_lbs")),
                "wins":         pick("wins",        row.get(f"{corner}_wins")),
                "losses":       pick("losses",      row.get(f"{corner}_losses")),
                "win_streak":   pick("win_streak",  row.get(f"{corner}_current_win_streak")),
                "lose_streak":  pick("lose_streak", row.get(f"{corner}_current_lose_streak")),
                "sig_str_acc":  pick("sig_str_acc", row.get(f"{corner}_avg_SIG_STR_pct")),
                "sig_str_def":  pick("sig_str_def", row.get(f"{corner}_avg_opp_SIG_STR_pct")),
                "td_acc":       pick("td_acc",      row.get(f"{corner}_avg_TD_pct")),
                "td_def":       pick("td_def",      row.get(f"{corner}_avg_opp_TD_pct")),
                "slpm":         pick("slpm",        row.get(f"{corner}_avg_SIG_STR_landed")),
                "sapm":         pick("sapm",        row.get(f"{corner}_avg_SIG_STR_att")),
                "td_avg":       pick("td_avg",      row.get(f"{corner}_avg_TD_landed")),
                "sub_avg":      pick("sub_avg",     row.get(f"{corner}_avg_SUB_ATT")),
                "ko_avg":       pick("ko_avg",      row.get(f"{corner}_avg_KD")),
                "finish_rate":  pick("finish_rate", None),
                "stance":       pick("stance",      row.get(f"{corner}_Stance")),
                "weight_class": pick("weight_class",row.get("weight_class")),
            }
    return fighters


def train(data_path: str = r"data\combined_data.csv"):
    print("📦 Loading data...")
    df = load_and_clean(data_path)

    print("⚙️  Engineering features (25 original + 7 new = 32 total)...")
    df = engineer_features(df)

    # Balance: flip every fight to remove red-corner bias
    df_flipped = df.copy()
    for col in FEATURE_COLS:
        if col in df_flipped.columns:
            df_flipped[col] = -df[col]
    df_flipped["label"] = 1 - df["label"]

    df_balanced = pd.concat([df, df_flipped], ignore_index=True)
    print(f"  Original: {len(df)} fights → Balanced: {len(df_balanced)} (2x flipped)")

    train_df, test_df = time_based_split(df_balanced, test_years=1)

    for col in FEATURE_COLS:
        if col not in train_df.columns:
            train_df[col] = 0
            test_df[col]  = 0

    X_train = train_df[FEATURE_COLS]
    y_train = train_df["label"]
    X_test  = test_df[FEATURE_COLS]
    y_test  = test_df["label"]

    print("\n🏋️  Building stacking ensemble...")
    pipeline = build_pipeline(X_train, y_train)

    print("🏋️  Training... (this may take 3-7 minutes)")
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    proba = pipeline.predict_proba(X_test)[:, 1]
    acc   = accuracy_score(y_test, preds)
    auc   = roc_auc_score(y_test, proba)

    print(f"\n✅ Test accuracy : {acc:.3f}  ({acc*100:.1f}%)")
    print(f"✅ ROC-AUC score : {auc:.3f}  (1.0 = perfect, 0.5 = random)")
    print(classification_report(y_test, preds, target_names=["Blue wins", "Red wins"]))

    joblib.dump(pipeline, MODEL_PATH)
    print(f"💾 Model saved → {MODEL_PATH}")

    if os.path.exists(FIGHTERS_PATH):
        fighters = joblib.load(FIGHTERS_PATH)
        print(f"ℹ️  Keeping fighter profiles → {FIGHTERS_PATH} ({len(fighters)} fighters)")
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

    # Style classification at prediction time
    r_style = classify_style(
        g(fighter_red, "slpm"), g(fighter_red, "sapm"),
        g(fighter_red, "td_avg"), g(fighter_red, "sub_avg"),
        g(fighter_red, "sig_str_acc")
    )
    b_style = classify_style(
        g(fighter_blue, "slpm"), g(fighter_blue, "sapm"),
        g(fighter_blue, "td_avg"), g(fighter_blue, "sub_avg"),
        g(fighter_blue, "sig_str_acc")
    )

    r_is_striker  = 1 if r_style == 1 else 0
    b_is_striker  = 1 if b_style == 1 else 0
    r_is_grappler = 1 if r_style in [2, 3] else 0
    b_is_grappler = 1 if b_style in [2, 3] else 0

    style_matchup = 0
    if r_is_striker and b_is_grappler:
        style_matchup = -1
    elif r_is_grappler and b_is_striker:
        style_matchup = 1

    r_win_streak  = g(fighter_red,  "win_streak")
    b_win_streak  = g(fighter_blue, "win_streak")
    r_lose_streak = g(fighter_red,  "lose_streak")
    b_lose_streak = g(fighter_blue, "lose_streak")

    r_form = r_win_streak - (r_lose_streak * 1.5)
    b_form = b_win_streak - (b_lose_streak * 1.5)

    r_slpm = g(fighter_red,  "slpm")
    b_slpm = g(fighter_blue, "slpm")
    r_sapm = g(fighter_red,  "sapm")
    b_sapm = g(fighter_blue, "sapm")

    features = {
        "age_diff":               g(fighter_red, "age")          - g(fighter_blue, "age"),
        "reach_diff":             g(fighter_red, "reach_cms")     - g(fighter_blue, "reach_cms"),
        "height_diff":            g(fighter_red, "height_cms")    - g(fighter_blue, "height_cms"),
        "weight_diff":            g(fighter_red, "weight_lbs")    - g(fighter_blue, "weight_lbs"),
        "win_rate_diff":          win_rate(fighter_red)           - win_rate(fighter_blue),
        "win_streak_diff":        r_win_streak                    - b_win_streak,
        "total_fights_diff":      total(fighter_red)              - total(fighter_blue),
        "finish_rate_diff":       g(fighter_red, "finish_rate")   - g(fighter_blue, "finish_rate"),
        "sig_str_acc_diff":       g(fighter_red, "sig_str_acc")   - g(fighter_blue, "sig_str_acc"),
        "sig_str_def_diff":       g(fighter_red, "sig_str_def")   - g(fighter_blue, "sig_str_def"),
        "slpm_diff":              r_slpm                          - b_slpm,
        "sapm_diff":              r_sapm                          - b_sapm,
        "td_acc_diff":            g(fighter_red, "td_acc")        - g(fighter_blue, "td_acc"),
        "td_def_diff":            g(fighter_red, "td_def")        - g(fighter_blue, "td_def"),
        "td_avg_diff":            g(fighter_red, "td_avg")        - g(fighter_blue, "td_avg"),
        "sub_avg_diff":           g(fighter_red, "sub_avg")       - g(fighter_blue, "sub_avg"),
        "ko_avg_diff":            g(fighter_red, "ko_avg")        - g(fighter_blue, "ko_avg"),
        "ko_rate_diff":           g(fighter_red, "ko_avg")        - g(fighter_blue, "ko_avg"),
        "sub_rate_diff":          g(fighter_red, "sub_avg")       - g(fighter_blue, "sub_avg"),
        "recent_win_rate_diff":   r_win_streak                    - b_win_streak,
        "days_since_fight_diff":  0.0,
        "experience_diff":        total(fighter_red)              - total(fighter_blue),
        "title_fight":            0.0,
        "is_main_event":          0.0,
        "avg_opponent_wins_diff": 0.0,
        # New features
        "style_matchup":          style_matchup,
        "R_is_striker":           r_is_striker,
        "B_is_striker":           b_is_striker,
        "R_is_grappler":          r_is_grappler,
        "B_is_grappler":          b_is_grappler,
        "recent_form_diff":       r_form - b_form,
        "trend_str_diff":         (r_win_streak / max(total(fighter_red), 1)) - win_rate(fighter_red) -
                                  ((b_win_streak / max(total(fighter_blue), 1)) - win_rate(fighter_blue)),
        "trend_td_diff":          0.0,
        "str_differential_diff":  (r_slpm - r_sapm) - (b_slpm - b_sapm),
        "inactivity_diff":        0.0,
        "weight_class_change_R":  0.0,
        "weight_class_change_B":  0.0,
        "home_advantage":         0.0,
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
    train(r"data\combined_data.csv")