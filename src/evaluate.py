import pathlib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
import torch


from model import SimpleClassifier


ROOT = pathlib.Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / 'models' / 'bytebrain.pt'
SCALER_PATH = ROOT / 'models' / 'scaler.joblib'




def evaluate_on_csv(csv_path: str, label_col: str = 'label'):
df = pd.read_csv(csv_path)
y = df[label_col].values.astype(np.float32)
X = df.drop(columns=[label_col]).values.astype(np.float32)


import joblib
scaler = joblib.load(SCALER_PATH)
X = scaler.transform(X).astype(np.float32)


model = SimpleClassifier(input_dim=X.shape[1])
state = torch.load(MODEL_PATH, map_location='cpu')
model.load_state_dict(state)
model.eval()


with torch.no_grad():
logits = model(torch.from_numpy(X))
probs = torch.sigmoid(logits).numpy()
preds = (probs > 0.5).astype(np.int32)


print("ROC-AUC:", roc_auc_score(y, probs))
print("Confusion Matrix:\n", confusion_matrix(y, preds))
print("\nReport:\n", classification_report(y, preds, digits=3))




if __name__ == "__main__":
import argparse
p = argparse.ArgumentParser()
p.add_argument('--csv', required=True, help='Path to CSV with features + label column')
p.add_argument('--label-col', default='label')
args = p.parse_args()
evaluate_on_csv(args.csv, args.label_col)