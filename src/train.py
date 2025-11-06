import os
import pathlib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
import joblib
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from torch.optim import Adam


from model import SimpleClassifier, count_parameters


ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'train.csv'
MODEL_DIR = ROOT / 'models'
MODEL_DIR.mkdir(parents=True, exist_ok=True)
SCALER_PATH = MODEL_DIR / 'scaler.joblib'
MODEL_PATH = MODEL_DIR / 'bytebrain.pt'
META_PATH = MODEL_DIR / 'meta.joblib'


RANDOM_STATE = 42
BATCH_SIZE = 128
EPOCHS = 15
LR = 1e-3




def load_or_make_data():
if DATA.exists():
df = pd.read_csv(DATA)
assert 'label' in df.columns, "Expected a 'label' column in data/train.csv"
y = df['label'].values.astype(np.float32)
X = df.drop(columns=['label']).values.astype(np.float32)
feature_names = [c for c in df.columns if c != 'label']
return X, y, feature_names
# generate synthetic data for a smooth first run
X, y = make_classification(
n_samples=4000, n_features=5, n_informative=4, n_redundant=0,
random_state=RANDOM_STATE, weights=[0.5, 0.5]
)
feature_names = [f"x{i}" for i in range(X.shape[1])]
return X.astype(np.float32), y.astype(np.float32), feature_names




def main():
X, y, feature_names = load_or_make_data()


X_train, X_val, y_train, y_val = train_test_split(
X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)


scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train).astype(np.float32)
X_val = scaler.transform(X_val).astype(np.float32)


joblib.dump(scaler, SCALER_PATH)


train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))


train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE)


main()