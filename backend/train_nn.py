"""
UFC Neural Network Meta-Learner
Trains a small neural network on top of the stacking ensemble predictions.
Run AFTER model.py has been trained.

Run: python train_nn.py
"""

import pandas as pd
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

# Import feature engineering from model.py
from model import (
    load_and_clean, engineer_features, time_based_split,
    tune_xgboost, FEATURE_COLS, FIGHTERS_PATH
)

MODEL_PATH    = "ufc_model.joblib"
NN_MODEL_PATH = "ufc_nn_model.joblib"


def tune_nn(X_train, y_train, n_trials=20):
    """Use Optuna to find best Neural Network architecture."""
    if not HAS_OPTUNA:
        return {
            "hidden_layer_sizes": (128, 64, 32),
            "activation": "relu",
            "alpha": 0.001,
            "learning_rate_init": 0.001,
        }

    print(f"  🔍 Tuning Neural Network with Optuna ({n_trials} trials)...")

    # Pre-impute and scale data for Optuna tuning
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    imputer = SimpleImputer(strategy="median")
    scaler  = StandardScaler()
    X_clean = imputer.fit_transform(X_train)
    X_clean = scaler.fit_transform(X_clean)

    def objective(trial):
        n_layers = trial.suggest_int("n_layers", 2, 4)
        layers   = tuple([
            trial.suggest_int(f"n_units_l{i}", 32, 256)
            for i in range(n_layers)
        ])
        params = {
            "hidden_layer_sizes": layers,
            "activation":         trial.suggest_categorical("activation", ["relu", "tanh"]),
            "alpha":              trial.suggest_float("alpha", 1e-5, 1e-1, log=True),
            "learning_rate_init": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "max_iter":           500,
            "random_state":       42,
            "early_stopping":     True,
            "validation_fraction": 0.1,
        }
        model  = MLPClassifier(**params)
        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(model, X_clean, y_train, cv=3, scoring="accuracy", n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    print(f"  ✅ Best NN accuracy (CV): {study.best_value:.3f}")
    return study.best_params


def build_nn_pipeline(X_train, y_train):
    """Build stacking ensemble with Neural Network as meta-learner."""

    rf = RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_split=10,
        random_state=42, n_jobs=-1, class_weight="balanced",
    )
    gb = GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42,
    )
    estimators = [("rf", rf), ("gb", gb)]

    if HAS_XGB:
        estimators.append(("xgb", xgb.XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            random_state=42, eval_metric="logloss",
        )))
        print("  ✅ XGBoost added")

    if HAS_LGB:
        estimators.append(("lgb", lgb.LGBMClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            num_leaves=31, random_state=42, verbose=-1,
        )))
        print("  ✅ LightGBM added")

    # Tune Neural Network meta-learner
    best_nn_params = tune_nn(X_train, y_train, n_trials=20)

    # Build NN meta-learner
    nn_meta = MLPClassifier(
        hidden_layer_sizes = best_nn_params.get("hidden_layer_sizes", (128, 64, 32)),
        activation         = best_nn_params.get("activation", "relu"),
        alpha              = best_nn_params.get("alpha", 0.001),
        learning_rate_init = best_nn_params.get("lr", 0.001),
        max_iter           = 1000,
        random_state       = 42,
        early_stopping     = True,
        validation_fraction= 0.1,
    )
    print(f"  ✅ Neural Network meta-learner: {nn_meta.hidden_layer_sizes}")

    # Stacking: base models → Neural Network meta-learner
    stacking = StackingClassifier(
        estimators=estimators,
        final_estimator=nn_meta,
        cv=5,
        stack_method="predict_proba",
        n_jobs=-1,
    )

    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("model",   stacking),
    ])


def train_nn(data_path: str = r"data\combined_data.csv"):
    print("📦 Loading data...")
    df = load_and_clean(data_path)

    print("⚙️  Engineering features...")
    df = engineer_features(df)

    # Balance
    df_flipped = df.copy()
    for col in FEATURE_COLS:
        if col in df_flipped.columns:
            df_flipped[col] = -df[col]
    df_flipped["label"] = 1 - df["label"]
    df_balanced = pd.concat([df, df_flipped], ignore_index=True)
    print(f"  Original: {len(df)} fights → Balanced: {len(df_balanced)}")

    # Time split
    train_df, test_df = time_based_split(df_balanced, test_years=1)

    for col in FEATURE_COLS:
        if col not in train_df.columns:
            train_df[col] = 0
            test_df[col]  = 0

    X_train = train_df[FEATURE_COLS]
    y_train = train_df["label"]
    X_test  = test_df[FEATURE_COLS]
    y_test  = test_df["label"]

    print("\n🧠 Building Neural Network stacking ensemble...")
    pipeline = build_nn_pipeline(X_train, y_train)

    print("🏋️  Training... (this may take 5-10 minutes)")
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    proba = pipeline.predict_proba(X_test)[:, 1]
    acc   = accuracy_score(y_test, preds)
    auc   = roc_auc_score(y_test, proba)

    print(f"\n✅ Neural Network Test accuracy : {acc:.3f}  ({acc*100:.1f}%)")
    print(f"✅ Neural Network ROC-AUC score : {auc:.3f}")
    print(classification_report(y_test, preds, target_names=["Blue wins", "Red wins"]))

    # Compare with existing model
    if os.path.exists(MODEL_PATH):
        existing = joblib.load(MODEL_PATH)
        existing_preds = existing.predict(
            pd.DataFrame([{col: 0 for col in FEATURE_COLS}]
        ))
        try:
            existing_acc = accuracy_score(y_test, existing.predict(X_test))
            print(f"\n📊 Comparison:")
            print(f"   Old model (LR meta):  {existing_acc*100:.1f}%")
            print(f"   New model (NN meta):  {acc*100:.1f}%")
            if acc > existing_acc:
                print(f"   ✅ Neural Network is BETTER by {(acc-existing_acc)*100:.1f}%")
                joblib.dump(pipeline, MODEL_PATH)
                print(f"   💾 Replaced model → {MODEL_PATH}")
            else:
                print(f"   ⚠️  Logistic Regression was better — keeping old model")
                joblib.dump(pipeline, NN_MODEL_PATH)
                print(f"   💾 Saved NN model anyway → {NN_MODEL_PATH}")
        except Exception as e:
            joblib.dump(pipeline, NN_MODEL_PATH)
            print(f"💾 NN model saved → {NN_MODEL_PATH}")
    else:
        joblib.dump(pipeline, MODEL_PATH)
        print(f"💾 Model saved → {MODEL_PATH}")


if __name__ == "__main__":
    train_nn(r"data\combined_data.csv")