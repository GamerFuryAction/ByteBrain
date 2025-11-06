from typing import List, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class ByteBrainChat:
"""Tiny local chat engine using a small causal LM.
- Default model: 'distilgpt2' (~300MB). For faster tests use 'sshleifer/tiny-gpt2'.
- Keeps a bounded in-memory conversation history per session_id.
"""
def __init__(self, model_name: str = "distilgpt2", system_prompt: str = "You are a concise, friendly assistant.", max_history: int = 6):
self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
self.tokenizer = AutoTokenizer.from_pretrained(model_name)
if self.tokenizer.pad_token_id is None:
self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
self.model = AutoModelForCausalLM.from_pretrained(model_name)
self.model.to(self.device).eval()
self.system = system_prompt
self.max_history = max_history
self.sessions = {} # session_id -> List[Tuple[role, text]]


def _compose_prompt(self, session_id: str, user_msg: str) -> str:
history: List[Tuple[str, str]] = self.sessions.get(session_id, [])[-self.max_history:]
parts = [f"### System: {self.system}"]
for role, text in history:
prefix = "User" if role == 'user' else 'Assistant'
parts.append(f"{prefix}: {text}")
parts.append(f"User: {user_msg}
Assistant:")
return "
".join(parts)


def chat(self, session_id: str, message: str, max_new_tokens: int = 128, temperature: float = 0.7, top_p: float = 0.9):
prompt = self._compose_prompt(session_id, message)
inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
with torch.no_grad():
output = self.model.generate(
**inputs,
max_new_tokens=max_new_tokens,
do_sample=True,
temperature=temperature,
top_p=top_p,
pad_token_id=self.tokenizer.eos_token_id,
eos_token_id=self.tokenizer.eos_token_id,
)
# Decode only the newly generated part
gen_ids = output[0][inputs['input_ids'].shape[-1]:]
text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
# Stop at next "User:" if the model invents another turn
reply = text.split("
User:")[0].strip()
# Update history
hist = self.sessions.setdefault(session_id, [])
hist.append(("user", message))
hist.append(("assistant", reply))
return reply, len(hist)


def reset(self, session_id: str):
self.sessions.pop(session_id, None)