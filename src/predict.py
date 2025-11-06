import pathlib
import numpy as np
import joblib
import torch
from model import SimpleClassifier


ROOT = pathlib.Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / 'models' / 'bytebrain.pt'
SCALER_PATH = ROOT / 'models' / 'scaler.joblib'
META_PATH = ROOT / 'models' / 'meta.joblib'




class ByteBrainPredictor:
def __init__(self):
meta = joblib.load(META_PATH)
self.feature_names = meta['feature_names']
self.scaler = joblib.load(SCALER_PATH)
self.model = SimpleClassifier(input_dim=len(self.feature_names))
self.model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
self.model.eval()


def _vectorize(self, features: dict) -> np.ndarray:
ordered = [features.get(name, 0.0) for name in self.feature_names]
X = np.array([ordered], dtype=np.float32)
X = self.scaler.transform(X).astype(np.float32)
return X


def predict(self, features: dict):
X = self._vectorize(features)
with torch.no_grad():
logits = self.model(torch.from_numpy(X))
prob = torch.sigmoid(logits).item()
label = int(prob > 0.5)
return {"probability": prob, "label": label}