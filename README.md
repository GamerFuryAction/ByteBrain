# ByteBrain 🧠


A from‑scratch Python AI starter you can train and deploy in minutes.


## Features
- PyTorch training loop with metrics (AUC/F1/Accuracy)
- Synthetic data fallback for zero‑setup runs
- FastAPI server with `/predict`
- Clean, minimal structure for real projects


## Quickstart
```bash
pip install -r requirements.txt
python src/train.py
uvicorn app.main:app --reload
