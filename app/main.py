from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Optional


# Lazy imports keep startup snappy even if model files are missing
try:
from src.predict import ByteBrainPredictor
from src.chat_model import ByteBrainChat
except Exception:
ByteBrainPredictor = None
ByteBrainChat = None


app = FastAPI(title="ByteBrain API", version="0.2.0")


# ==== Schemas ====
class PredictRequest(BaseModel):
features: Dict[str, float]


class ChatRequest(BaseModel):
session_id: str
message: str
max_new_tokens: Optional[int] = 128
temperature: Optional[float] = 0.7
top_p: Optional[float] = 0.9


# ==== Startup ====
@app.on_event("startup")
def _load_components():
global predictor, chat
if ByteBrainPredictor is None or ByteBrainChat is None:
raise RuntimeError("ByteBrain modules not available. Did you install requirements?")
# Tabular predictor (optional)
try:
predictor = ByteBrainPredictor()
except Exception:
predictor = None # ok if not trained yet
# Conversational model
chat = ByteBrainChat() # loads a small local LLM (distilgpt2 by default)


@app.get("/health")
def health():
return {"ok": True, "has_predictor": predictor is not None, "chat_loaded": chat is not None}


# ==== Predict (numeric features) ====
@app.post("/predict")
def predict(req: PredictRequest):
if predictor is None:
raise HTTPException(status_code=400, detail="Predictor not ready. Train the model first (python src/train.py)")
try:
result = predictor.predict(req.features)
return {"ok": True, **result}
except FileNotFoundError:
raise HTTPException(status_code=400, detail="Model not found. Train the model first (python src/train.py)")
except Exception as e:
raise HTTPException(status_code=500, detail=str(e))


# ==== Chat (LLM) ====
@app.post("/chat")
def chat_endpoint(req: ChatRequest):
try:
reply, history_len = chat.chat(
session_id=req.session_id,
message=req.message,
max_new_tokens=req.max_new_tokens,
temperature=req.temperature,
return {"ok": True}