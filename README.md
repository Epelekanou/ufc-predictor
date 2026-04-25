# 🥊 UFC Fight Predictor

A full-stack machine learning web application that predicts UFC fight outcomes using real fighter statistics. Select any two fighters and get an ML-powered prediction with detailed stats comparison.
->If its not working give it some time to start around 3-5 minutes.

🔗 **Live Demo:** https://ufc-predictor-32k6.vercel.app/

---

## 📸 Preview

> Pick any two UFC fighters → Get win probability + full stats breakdown

---

## 🧠 How It Works

1. **Data Collection** — Custom parallel web scraper pulls live fighter stats from [ufcstats.com](http://www.ufcstats.com) (4,400+ fighters)
2. **Feature Engineering** — Computes differential features between fighters (reach advantage, strike accuracy gap, win rate difference, etc.)
3. **ML Model** — Ensemble of Random Forest + Gradient Boosting trained on 6,000+ real UFC fights
4. **Bias Removal** — Training data is flipped (red/blue corner swapped) to eliminate corner-position bias
5. **Prediction** — Returns win probability for each fighter based on 19 statistical features

---

## 🏗️ Architecture

```
ufcstats.com
     ↓ (scraper_fast.py - parallel scraping)
fighters.joblib (4,475 fighters)
     ↓
FastAPI Backend ──→ ML Model (Random Forest + Gradient Boosting)
     ↓
React Frontend (Vercel) ←──→ Backend API (Render)
```

---

## ⚙️ Tech Stack

### Backend
| Technology | Purpose |
|---|---|
| Python | Core language |
| FastAPI | REST API framework |
| scikit-learn | ML model (Random Forest + Gradient Boosting ensemble) |
| pandas / numpy | Data processing & feature engineering |
| BeautifulSoup + requests | Parallel web scraping |
| joblib | Model serialization |

### Frontend
| Technology | Purpose |
|---|---|
| React | UI framework |
| Recharts | Radar chart & bar chart visualizations |
| Axios | API communication |
| CSS3 | Custom dark theme styling |

### Deployment
| Service | Purpose |
|---|---|
| Render | Backend API hosting |
| Vercel | Frontend hosting |
| GitHub | Version control & CI/CD |

---

## 📊 ML Model Details

- **Algorithm:** Voting Ensemble (Random Forest + Gradient Boosting)
- **Training data:** 6,012 UFC fights (Kaggle dataset)
- **Accuracy:** ~63% (comparable to professional oddsmakers at 65-70%)
- **Features used (19 total):**
  - Physical: age, reach, height, weight differentials
  - Record: win rate, win streak, total fights, finish rate
  - Striking: sig. strike accuracy, defense, output (SLpM), absorption (SApM)
  - Grappling: takedown accuracy, defense, average, submission attempts
  - Finishing: KO rate, submission rate

---

## 🚀 Run Locally

### Prerequisites
- Python 3.10+
- Node.js 18+

### Backend
```bash
cd backend
pip install -r requirements.txt
python model.py          # trains the model
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
│   ├── model.py             # ML training & prediction logic
│   ├── scraper_fast.py      # Parallel web scraper (ufcstats.com)
│   ├── fighters.joblib      # Scraped fighter profiles (4,475 fighters)
│   ├── ufc_model.joblib     # Trained ML model
│   └── requirements.txt
└── frontend/
    ├── src/
    │   ├── App.jsx
    │   └── components/
    │       ├── FighterSelect.jsx    # Search & autocomplete
    │       ├── FighterCard.jsx      # Fighter stats display
    │       ├── PredictionResult.jsx # Charts & prediction
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

---

## 👤 Author

**Evangelos** — [GitHub](https://github.com/Epelekanou)

---

## 📄 Data Sources

- Fight history: [Kaggle UFC Dataset](https://www.kaggle.com/datasets/rajeevw/ufcdata)
- Live fighter stats: [UFC Stats](http://www.ufcstats.com)
