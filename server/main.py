from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from livekit import api as lkapi
from pydantic import BaseModel

from server.app import log_step, logger, settings
from server.app.agent import LKAgent
from server.rag.simple_rag import SimpleRAG

RAG = SimpleRAG(corpus_dir=str(settings.rag_corpus_dir))
if not settings.rag_corpus_dir.exists():
    logger.warning("RAG corpus dir not found: %s", settings.rag_corpus_dir)
else:
    logger.info("RAG corpus dir: %s", settings.rag_corpus_dir)

app = FastAPI(title="Jarvis · LiveKit × Sarvam")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODE_STATE: Dict[str, str] = {"mode": "conversation"}
AGENT: Optional[LKAgent] = None


class ModeBody(BaseModel):
    mode: str


class ClientLogBody(BaseModel):
    action: str
    details: Optional[Dict[str, Any]] = None


def mint_token(identity: str, room: str) -> str:
    grant = lkapi.VideoGrants(
        room_join=True,
        room=room,
        can_publish=True,
        can_publish_data=True,
        can_subscribe=True,
    )
    return (
        lkapi.AccessToken(settings.livekit_api_key, settings.livekit_api_secret)
        .with_identity(identity)
        .with_grants(grant)
        .to_jwt()
    )


@app.get("/health")
async def health():
    return {"ok": True}


@app.get("/token")
async def get_token(identity: Optional[str] = None, room: Optional[str] = None):
    identity = identity or f"web-{int(asyncio.get_running_loop().time() * 1000)}"
    room = room or settings.room_name
    token = mint_token(identity, room)
    return {"identity": identity, "room": room, "token": token, "livekit_url": settings.livekit_url}


@app.get("/mode")
async def get_mode():
    return MODE_STATE


@app.post("/mode")
async def post_mode(body: ModeBody):
    mode = body.mode.strip().lower()
    if mode not in {"conversation", "push-to-talk"}:
        raise HTTPException(status_code=400, detail="invalid mode")
    MODE_STATE["mode"] = mode
    log_step("mode.update", "success", {"mode": mode})
    return MODE_STATE


@app.post("/client-log")
async def client_log(body: ClientLogBody):
    log_step("client.log", "info", {"action": body.action, "details": body.details})
    return {"ok": True}


@app.get("/diagnostics")
async def diagnostics():
    if AGENT is None:
        return {"mode": MODE_STATE.get("mode"), "status": {"connected": False, "state": "booting"}}
    return {"mode": MODE_STATE.get("mode"), "status": AGENT.diagnostics()}


@app.post("/flush")
async def flush():
    if AGENT is None:
        return {"ok": True, "status": "agent-unavailable"}
    status = await AGENT.force_flush()
    return {"ok": True, "status": status}


@app.on_event("startup")
async def on_startup():
    global AGENT
    try:
        settings.validate()
    except ValueError as exc:
        logger.error("Configuration error: %s", exc)
        raise

    token = mint_token(settings.agent_identity, settings.room_name)
    try:
        import jwt as pyjwt  # type: ignore

        claims = pyjwt.decode(token, options={"verify_signature": False})
        logger.info("Agent identity=%s grants=%s", settings.agent_identity, json.dumps(claims.get("video", {})))
    except Exception:
        pass

    AGENT = LKAgent(settings=settings, rag=RAG, token=token)
    asyncio.create_task(AGENT.connect())


@app.on_event("shutdown")
async def on_shutdown():
    if AGENT:
        await AGENT.aclose()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
