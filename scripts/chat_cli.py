import argparse
import requests


parser = argparse.ArgumentParser()
parser.add_argument('--host', default='http://127.0.0.1:8000')
parser.add_argument('--session', default='local')
args = parser.parse_args()


print("ByteBrain Chat — type your message, Ctrl+C to exit")
while True:
try:
msg = input('You: ').strip()
if not msg:
continue
r = requests.post(f"{args.host}/chat", json={
"session_id": args.session,
"message": msg
})
r.raise_for_status()
print('AI :', r.json().get('reply'))
except (KeyboardInterrupt, EOFError):
print("
Bye!")
break