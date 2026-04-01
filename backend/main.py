from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from anthropic import Anthropic
import os
from dotenv import load_dotenv
from model import load_model, load_fighters, predict_fight
import math 
load_dotenv()

app = FastAPI(title="UFC Fight Predictor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load ML model + fighter profiles at startup
model    = load_model()
fighters = load_fighters()
claude   = Anthropic()  # reads ANTHROPIC_API_KEY from .env


# ── Schemas ───────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    red_fighter:  str
    blue_fighter: str


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/fighters/search")
def search_fighters(q: str = ""):
    """Return fighter names matching the query string."""
    q_lower = q.lower()
    results = [
        name for name in fighters
        if q_lower in name.lower()
    ]
    return sorted(results)[:20]


@app.get("/fighters/{name:path}")
def get_fighter(name: str):
    if name not in fighters:
        raise HTTPException(404, f"Fighter '{name}' not found")
    # Clean NaN values so JSON serialization works
    clean = {
        k: (None if isinstance(v, float) and math.isnan(v) else v)
        for k, v in fighters[name].items()
    }
    return clean

@app.post("/predict")
def predict(req: PredictRequest):
    red  = fighters.get(req.red_fighter)
    blue = fighters.get(req.blue_fighter)

    if not red:
        raise HTTPException(404, f"Fighter '{req.red_fighter}' not found")
    if not blue:
        raise HTTPException(404, f"Fighter '{req.blue_fighter}' not found")

    try:
        def clean(f):
            return {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in f.items()}

        result = predict_fight(red, blue, model)
        return {
            "red_fighter":  clean(red),
            "blue_fighter": clean(blue),
            **result,
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"Prediction error: {str(e)}")
 

@app.post("/analyze")
def analyze(req: PredictRequest):
    """Ask Claude to explain the matchup in natural language."""
    red  = fighters.get(req.red_fighter)
    blue = fighters.get(req.blue_fighter)

    if not red:
        raise HTTPException(404, f"Fighter '{req.red_fighter}' not found")
    if not blue:
        raise HTTPException(404, f"Fighter '{req.blue_fighter}' not found")

    prediction = predict_fight(red, blue, model)

    prompt = f"""You are an expert UFC analyst. Analyze this matchup and give a detailed breakdown.

FIGHTER 1 (Red corner): {red['name']}
- Record: {red.get('wins', '?')}-{red.get('losses', '?')}
- Age: {red.get('age', '?')}
- Height: {red.get('height_cms', '?')} cm | Reach: {red.get('reach_cms', '?')} cm
- Win streak: {red.get('win_streak', '?')}
- Sig. strike accuracy: {red.get('sig_str_acc', '?')}
- Takedown accuracy: {red.get('td_acc', '?')}
- Avg knockdowns per fight: {red.get('ko_avg', '?')}
- Avg submission attempts: {red.get('sub_avg', '?')}
- Stance: {red.get('stance', '?')}

FIGHTER 2 (Blue corner): {blue['name']}
- Record: {blue.get('wins', '?')}-{blue.get('losses', '?')}
- Age: {blue.get('age', '?')}
- Height: {blue.get('height_cms', '?')} cm | Reach: {blue.get('reach_cms', '?')} cm
- Win streak: {blue.get('win_streak', '?')}
- Sig. strike accuracy: {blue.get('sig_str_acc', '?')}
- Takedown accuracy: {blue.get('td_acc', '?')}
- Avg knockdowns per fight: {blue.get('ko_avg', '?')}
- Avg submission attempts: {blue.get('sub_avg', '?')}
- Stance: {blue.get('stance', '?')}

ML MODEL PREDICTION: {prediction['winner']} wins
- {red['name']} win probability: {prediction['red_confidence']}%
- {blue['name']} win probability: {prediction['blue_confidence']}%

Give a 3-4 paragraph analysis covering:
1. Striking matchup — who has the edge and why
2. Grappling & wrestling matchup
3. Key advantages / disadvantages for each fighter
4. Your predicted outcome and how it likely ends (KO, submission, or decision)

Be confident, specific, and use the stats to back up your points. Write like a real fight analyst.
"""

    response = claude.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=800,
        messages=[{"role": "user", "content": prompt}],
    )

    return {"analysis": response.content[0].text}


@app.get("/health")
def health():
    return {"status": "ok", "fighters_loaded": len(fighters)}