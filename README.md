# 🥊 UFC Fight Predictor

A full-stack machine learning web application that predicts UFC fight outcomes using real fighter statistics. Select any two fighters and get an ML-powered prediction with detailed stats comparison, confidence scores, radar charts, and fighter style classification.

> ⚠️ First load may take 30–60 seconds — the backend spins down after inactivity on Render's free tier.

🔗 **Live Demo:** https://ufc-predictor-32k6.vercel.app/

---

## 🧠 How It Works

1. **Fighter Scraping** — Custom parallel scraper pulls live fighter stats from [ufcstats.com](http://www.ufcstats.com) (4,480+ fighters, 10 threads via ThreadPoolExecutor)
2. **Fight Scraping** — Second scraper collects post-2021 fight results (217 events, 2,394 new fights) and merges with Kaggle dataset
3. **Feature Engineering** — Computes 32 differential features between fighters across 7 categories
4. **Style Classification** — Each fighter is automatically classified as Striker 🥊 / Grappler 🔒 / Wrestler 💪 / Balanced ⚖️ based on their real UFC stats
5. **ML Model** — Stacking Ensemble (Random Forest + Gradient Boosting + XGBoost + LightGBM) with Logistic Regression meta-learner
6. **Hyperparameter Tuning** — XGBoost parameters auto-tuned using Optuna (30 trials)
7. **Bias Removal** — Training data is flipped (red/blue corner swapped) to eliminate corner-position bias
8. **Time-based Validation** — Trained on older fights, tested on most recent year for realistic evaluation

---

## 🏗️ Architecture

```
ufcstats.com
     ↓ (scraper_fast.py — parallel fighter scraping, 10 threads)
fighters.joblib (4,480 fighters)

ufcstats.com
     ↓ (scrape_new_fights.py — 217 events, 2,394 fights post-2021)
data/combined_data.csv (8,405 total fights)
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
| Optuna | Hyperparameter tuning (30 trials) |
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
- **Training data:** 8,405 UFC fights (Kaggle dataset 1994–2021 + custom scraped 2021–2026)
- **Accuracy:** ~80.4% (time-based validation)
- **ROC-AUC:** 0.880
- **Validation strategy:** Time-based split — train on older fights, test on most recent year
- **Features used (32 total):**

| Category | Features |
|---|---|
| Physical | Age, reach, height, weight differentials |
| Record | Win rate, win streak, total fights, finish rate |
| Striking | Sig. strike accuracy, defense, SLpM, SApM |
| Grappling | TD accuracy, defense, average, submission attempts |
| Finishing | KO rate, submission rate |
| Form & Trend | Recent form, performance trend, experience gap |
| Style | Fighter style classification, style matchup score |
| Context | Title fight flag, inactivity penalty, weight class change, home advantage |

---

## 🎨 Fighter Style Classification

Each fighter is automatically classified based on their real UFC statistics:

| Style | Icon | Criteria |
|---|---|---|
| **Striker** | 🥊 | High SLpM + high accuracy, low takedown activity |
| **Grappler** | 🔒 | High submission attempts + takedown average |
| **Wrestler** | 💪 | High takedown average, low submission attempts |
| **Balanced** | ⚖️ | Mixed stats across all categories |

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
python scraper_fast.py       # scrape fighter profiles (~10 min)
python scrape_new_fights.py  # scrape post-2021 fights (~30-40 min)
python model.py              # train the model (~5 min)
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

## 🕷️ Update Fighter & Fight Data

```bash
cd backend
python scraper_fast.py       # updates fighter profiles
python scrape_new_fights.py  # scrapes latest fights
python model.py              # retrains model with fresh data
```

---

## 📁 Project Structure

```
ufc-predictor/
├── backend/
│   ├── main.py                  # FastAPI app & API routes
│   ├── model.py                 # ML training & prediction (32 features, stacking ensemble)
│   ├── train_nn.py              # Neural network meta-learner experiment
│   ├── scraper_fast.py          # Parallel fighter scraper (4,480 fighters)
│   ├── scrape_new_fights.py     # Fight scraper (2,394 post-2021 fights)
│   ├── fighters.joblib          # Scraped fighter profiles
│   ├── ufc_model.joblib         # Trained stacking ensemble model
│   └── requirements.txt
└── frontend/
    ├── src/
    │   ├── App.jsx
    │   ├── App.css                  # Dark theme, mobile-responsive
    │   └── components/
    │       ├── FighterSelect.jsx    # Search & autocomplete
    │       ├── FighterCard.jsx      # Fighter stats + style badge
    │       └── PredictionResult.jsx # Charts & prediction output
    └── package.json
```

---

## 🔮 Features

- 🔍 **Fighter search** — autocomplete from 4,480+ UFC fighters
- 🥊 **Style classification** — automatic Striker / Grappler / Wrestler / Balanced badge
- 📊 **Stats comparison** — striking, grappling, physical, finishing stats
- 🎯 **Win probability** — confidence bar showing each fighter's chances
- 📡 **Radar chart** — 8-dimension visual profile comparison
- 📈 **Head-to-head bar chart** — direct stat comparisons
- ✅ **Advantage cards** — automatically highlights each fighter's edges
- 📱 **Mobile responsive** — fighters shown side by side on all screen sizes

---

## 📈 Model Evolution

| Version | Dataset | Features | Accuracy | ROC-AUC |
|---|---|---|---|---|
| v1 — Basic Ensemble | 6,012 fights | 19 | 63% | 0.663 |
| v2 — New Data | 8,405 fights | 25 | 83.7% | 0.915 |
| v3 — Full Features | 8,405 fights | 32 | **80.4%** | **0.880** |

---

## 👤 Author

**Evangelos Pelekanou** — [GitHub](https://github.com/Epelekanou)

---

## 📄 Data Sources

- Fight history: [Kaggle UFC Dataset](https://www.kaggle.com/datasets/rajeevw/ufcdata)
- Live fighter stats & new fights: [UFC Stats](http://www.ufcstats.com)
