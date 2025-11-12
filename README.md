# Jarvis · LiveKit × Sarvam · RAG

Realtime voice assistant that listens to your LiveKit room, detects end-of-speech, pushes the chunk through Sarvam STT → RAG → Sarvam Chat → Sarvam TTS, and streams the answer back over the same LiveKit data channels. This repo is structured so it can be pushed directly to [ayushcody/Jarvis](https://github.com/ayushcody/Jarvis).

## Project layout

```
.
├─ client/                 # Vanilla HTML/JS dashboard + status card + mic controls
├─ server/
│  ├─ app/
│  │  ├─ agent.py          # LKAgent: LiveKit subscriber + Sarvam pipeline
│  │  ├─ config.py         # Settings loader (.env → dataclass)
│  │  └─ logging.py        # Structured logging helper (log_step)
│  ├─ rag/                 # Tiny TF‑IDF retriever + sample corpus
│  ├─ util_audio.py        # PCM resampling + simple SilenceVAD helper
│  ├─ main.py              # FastAPI entrypoint + LiveKit token server
│  └─ requirements.txt
└─ README.md
```

## Requirements

- Python 3.10+
- Node/npm (optional – the client is static, so `python -m http.server` works)
- LiveKit project (Cloud or self-hosted) with API key/secret
- Sarvam API key (STT/LLM/TTS)

## Setup

### 1. Clone & install

```bash
git clone https://github.com/ayushcody/Jarvis.git
cd Jarvis/server
python -m venv .venv
source .venv/bin/activate             # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment

Create `server/.env` (not checked into git):

```
LIVEKIT_URL=wss://<project>.livekit.cloud
LIVEKIT_API_KEY=lk_apikey
LIVEKIT_API_SECRET=lk_secret
ROOM_NAME=demo-room

SARVAM_API_KEY=svm_key
SARVAM_LANGUAGE_CODE=en-IN
SARVAM_TTS_VOICE=bulbul:v2
SARVAM_STT_TRANSPORT=ws          # ws or rest
SARVAM_TTS_TRANSPORT=ws          # ws or rest

CORS_ALLOW_ORIGINS=http://localhost:4173
```

Optional knobs:

- `AGENT_IDENTITY` (defaults to `rag-agent`)
- `RAG_CORPUS_DIR` (defaults to `server/rag/corpus`)

### 3. Run the backend

```bash
cd server
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Endpoints:

- `GET /token` – returns (identity, room, token, livekit_url) for the browser.
- `GET/POST /mode` – persist conversation vs push-to-talk.
- `POST /client-log` – client telemetry (used for diagnostics timeline).
- `GET /diagnostics` – LiveKit connection flag + pipeline stage + last transcript/reply/TTS bytes.
- `POST /flush` – manually trigger a buffer flush (for debugging).

### 4. Serve the client

```bash
cd client
python3 -m http.server 4173
```

Open `http://localhost:4173`, set:

1. Token server URL → `http://localhost:8000`
2. Room ID → same as `ROOM_NAME`
3. Click **Connect & Publish**, allow the mic, and start talking.

When the orb is green, Jarvis is listening. When it switches to orange, Sarvam TTS is speaking back. The diagnostics card shows LiveKit status plus STT/RAG/Chat/TTS stages in real time.

## Development tips

- All structured telemetry flows through `log_step()` in `server/app/logging.py`. Use it for timeline-style logs.
- `LKAgent` lives in `server/app/agent.py`. The watchdog flush (`_watchdog_loop`) ensures every buffered chunk is sent even if VAD misses an end-of-speech event.
- The client polls `/diagnostics` every 4 s to update the card. If you add new pipeline stages, report them via `diag["stage"]` for immediate UI feedback.
- `server/.gitignore` blocks `.env`, `.venv`, node_modules, logs, CDN source maps, etc., so the repo stays clean when you push to GitHub.

## License

MIT – see the repository once it’s pushed to GitHub (`ayushcody/Jarvis`). Feel free to adapt for demos or production, just rotate secrets before deploying.
