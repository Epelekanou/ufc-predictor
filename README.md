# 🥊 UFC Fight Predictor

A full-stack machine learning web application that predicts UFC fight outcomes using real fighter statistics. Select any two fighters and get an ML-powered prediction with detailed stats comparison, confidence scores, and visual breakdowns.

> ⚠️ First load may take 30–60 seconds — the backend spins down after inactivity on Render's free tier.

🔗 **Live Demo:** https://ufc-predictor-32k6.vercel.app/

---

## 📸 Preview

> Pick any two UFC fighters → Get win probability + full stats breakdown + radar chart comparison

---

## 🧠 How It Works

1. **Data Collection** — Custom parallel web scraper pulls live fighter stats from [ufcstats.com](http://www.ufcstats.com) (4,400+ fighters, ~10x faster than sequential scraping using ThreadPoolExecutor)
2. **Feature Engineering** — Computes 25 differential features between fighters (reach advantage, strike accuracy gap, win rate difference, recent form, etc.)
3. **ML Model** — Stacking Ensemble (Random Forest + Gradient Boosting + XGBoost + LightGBM) with Logistic Regression as the meta-learner
4. **Hyperparameter Tuning** — XGBoost parameters auto-tuned using Optuna (30 trials)
5. **Bias Removal** — Training data is flipped (red/blue corner swapped) to eliminate corner-position bias
6. **Time-based Validation** — Trained on older fights, tested on recent ones for realistic evaluation
7. **Prediction** — Returns win probability for each fighter based on 25 statistical features

---

## 🏗️ Architecture

```
ufcstats.com
     ↓ (scraper_fast.py — parallel scraping, 10 threads)
fighters.joblib (4,475 fighters)
     ↓
FastAPI Backend ──→ Stacking Ensemble (RF + GBM + XGB + LGB → LR)
     ↓
React Frontend (Vercel) ←──→ Backend API (Render)
```

---

## ⚙️ Tech Stack

### Backend
| Technology | Purpose |
|---|---|
| Python 3.12 | Core language |
| FastAPI | REST API framework |
| scikit-learn | ML pipeline, Random Forest, Gradient Boosting, Stacking |
| XGBoost | Gradient boosting (Optuna-tuned) |
| LightGBM | Fast gradient boosting |
| Optuna | Hyperparameter tuning |
| pandas / numpy | Data processing & feature engineering |
| BeautifulSoup + requests | Parallel web scraping |
| joblib | Model & fighter data serialization |

### Frontend
| Technology | Purpose |
|---|---|
| React 19 | UI framework |
| Recharts | Radar chart & bar chart visualizations |
| Axios | API communication |
| CSS3 | Custom dark theme, mobile-responsive layout |

### Deployment
| Service | Purpose |
|---|---|
| Render | Backend API hosting (free tier) |
| Vercel | Frontend hosting (free tier) |
| GitHub | Version control & CI/CD |

---

## 📊 ML Model Details

- **Algorithm:** Stacking Ensemble — Random Forest + Gradient Boosting + XGBoost + LightGBM → Logistic Regression meta-learner
- **Training data:** 6,012 UFC fights (Kaggle dataset, Kaggle + ufcstats.com)
- **Accuracy:** ~62% (time-based validation — comparable to professional oddsmakers at 65–70%)
- **ROC-AUC:** 0.663
- **Validation strategy:** Time-based split (train on older fights, test on most recent year) — more realistic than random split
- **Features used (25 total):**
  - Physical: age, reach, height, weight differentials
  - Record: win rate, win streak, total fights, finish rate
  - Striking: sig. strike accuracy, defense, output (SLpM), absorption (SApM)
  - Grappling: takedown accuracy, defense, average, submission attempts
  - Finishing: KO rate, submission rate
  - New: recent form, experience gap, title fight flag, opponent quality proxy

---

## 🚀 Run Locally

### Prerequisites
- Python 3.10+
- Node.js 18+

### Backend
```bash
cd backend
pip install -r requirements.txt
pip install xgboost lightgbm optuna
python model.py          # trains the model (~3-5 min)
uvicorn main:app --reload
```

### Frontend
```bash
cd frontend
npm install
npm start
```

App runs at **http://localhost:3000**

---

## 🕷️ Update Fighter Data

To refresh fighter stats from ufcstats.com:
```bash
cd backend
python scraper_fast.py   # ~10 minutes, parallel scraping
python model.py          # retrain with fresh data
```

---

## 📁 Project Structure

```
ufc-predictor/
├── backend/
│   ├── main.py              # FastAPI app & API routes
│   ├── model.py             # ML training & prediction logic (stacking ensemble)
│   ├── scraper_fast.py      # Parallel web scraper (ufcstats.com, 10 threads)
│   ├── fighters.joblib      # Scraped fighter profiles (4,475 fighters)
│   ├── ufc_model.joblib     # Trained stacking ensemble model
│   └── requirements.txt
└── frontend/
    ├── src/
    │   ├── App.jsx
    │   ├── App.css              # Dark theme, mobile-responsive
    │   └── components/
    │       ├── FighterSelect.jsx    # Search & autocomplete
    │       ├── FighterCard.jsx      # Fighter stats display
    │       └── PredictionResult.jsx # Charts & prediction output
    └── package.json
```

---

## 🔮 Features

- 🔍 **Fighter search** — autocomplete from 4,400+ UFC fighters
- 📊 **Stats comparison** — striking, grappling, physical, finishing stats
- 🎯 **Win probability** — confidence bar showing each fighter's chances
- 📡 **Radar chart** — 8-dimension visual profile comparison
- 📈 **Head-to-head bar chart** — direct stat comparisons
- ✅ **Advantage cards** — automatically highlights each fighter's edges
- 📱 **Mobile responsive** — fighters shown side by side on all screen sizes

---

## 👤 Author

**Evangelos Pelekanou** — [GitHub](https://github.com/Epelekanou)

---

## 📄 Data Sources

- Fight history: [Kaggle UFC Dataset](https://www.kaggle.com/datasets/rajeevw/ufcdata)
- Live fighter stats: [UFC Stats](http://www.ufcstats.com)
