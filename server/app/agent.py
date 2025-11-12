from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import time
from typing import Any, Dict, Optional

import httpx
import websockets
from livekit import rtc

from server.rag.simple_rag import SimpleRAG
from server.util_audio import SilenceVAD, resample_to_16k_mono

from .config import Settings
from .logging import log_step

try:
    from sarvamai import AsyncSarvamAI, SarvamAI
except ImportError:  # pragma: no cover
    SarvamAI = None
    AsyncSarvamAI = None


def _normalize_conn_state(state: Any) -> tuple[bool, str, Optional[int]]:
    name = getattr(state, "name", None)
    value = getattr(state, "value", None)
    raw = state
    s = str(name if name is not None else raw if raw is not None else "").lower()

    numeric = None
    if isinstance(raw, (int, float)):
        numeric = int(raw)
    elif value is not None:
        try:
            numeric = int(value)
        except Exception:
            numeric = None
    elif s.isdigit():
        numeric = int(s)

    is_connected = False
    if s in {"connected", "connection_state_connected", "1"}:
        is_connected = True
    if numeric == 1:
        is_connected = True

    return is_connected, s or str(numeric or ""), numeric


class LKAgent:
    MAX_DATA_CHUNK = 10_000  # bytes

    def __init__(self, settings: Settings, rag: SimpleRAG, token: str) -> None:
        self.settings = settings
        self.rag = rag
        self.token = token
        self.url = settings.livekit_url

        self.room: Optional[rtc.Room] = None
        self.http = httpx.AsyncClient(timeout=30.0)

        try:
            self.vad = SilenceVAD(sample_rate=16000)
        except TypeError:
            try:
                self.vad = SilenceVAD(16000)
            except TypeError:
                self.vad = SilenceVAD()

        self.buffer = bytearray()
        self.buffer_lock = asyncio.Lock()
        self.logger = logging.getLogger(f"lk-agent[{settings.agent_identity}]")

        self.diag: Dict[str, Any] = {
            "connected": False,
            "state": "unknown",
            "state_value": None,
            "last_transcript": None,
            "stage": "idle",
            "last_reply_chars": 0,
            "last_tts_bytes": 0,
        }
        self.last_transcript: Optional[str] = None

        self.sarvam_sync = (
            SarvamAI(api_subscription_key=settings.sarvam_api_key) if settings.sarvam_api_key and SarvamAI else None
        )
        self.sarvam_async = (
            AsyncSarvamAI(api_subscription_key=settings.sarvam_api_key)
            if settings.sarvam_api_key and AsyncSarvamAI
            else None
        )

        self.watchdog_task: Optional[asyncio.Task] = None
        self.last_audio_ts_ms = int(time.time() * 1000)
        self.idle_flush_ms = 1200

    async def connect(self) -> None:
        log_step("agent.connect.start", "running", {"identity": self.settings.agent_identity})
        self.room = rtc.Room()

        @self.room.on("connection_state_changed")
        def _on_conn_state(state):
            is_connected, name_str, value = _normalize_conn_state(state)
            self.logger.info("Room connection state: %s (value=%s, connected=%s)", state, value, is_connected)
            self.diag.update({"connected": is_connected, "state": name_str, "state_value": value})
            log_step("agent.connection_state", "update", {"state": name_str, "value": value, "connected": is_connected})
            if is_connected:
                asyncio.create_task(self._prime_publisher_dc())

        @self.room.on("track_published")
        def _on_track_published(publication, participant):
            try:
                kind = getattr(publication, "kind", None)
                if kind == getattr(rtc.TrackKind, "KIND_AUDIO", None) or str(kind).lower() == "audio":
                    self.logger.info(
                        "Remote audio track published by %s — forcing subscribe",
                        getattr(participant, "identity", "unknown"),
                    )
                    log_step(
                        "agent.track_published",
                        "received",
                        {"participant": getattr(participant, "identity", "unknown")},
                    )
                    asyncio.create_task(self._ensure_subscription(publication))
            except Exception as exc:  # pragma: no cover
                self.report_error("track_published", exc)

        @self.room.on("track_subscribed")
        def _on_track_subscribed(track, publication, participant):
            try:
                kind = getattr(track, "kind", None)
                if str(kind).lower() not in ("audio", "kindaudio"):
                    return
            except Exception:
                pass
            self.logger.info(
                '{"step":"agent.track_subscribed","status":"received","details":{"participant":"%s"}}',
                getattr(participant, "identity", "unknown"),
            )
            asyncio.create_task(self.consume_audio(track))

        try:
            await self.room.connect(self.url, token=self.token)
            log_step("agent.connect.joined", "success", {"room": self.settings.room_name})
            await self._prime_publisher_dc()
            log_step("agent.publisher_dc", "prime_ping_sent", None)
            if not self.watchdog_task:
                self.watchdog_task = asyncio.create_task(self._watchdog_loop())
        except Exception as exc:
            log_step("agent.connect", "failed", {"error": repr(exc)})
            raise

        self.logger.info("Connected. Waiting for audio...")

    async def _prime_publisher_dc(self):
        assert self.room is not None
        try:
            await self.room.local_participant.publish_data(b"prime", reliable=True, topic="prime")
            self.logger.info("Prime data ping sent (publisher DC established).")
            log_step("agent.publisher_dc", "prime_ping_sent", None)
        except Exception as exc:
            self.report_error("prime-publish", exc)
            raise

    async def consume_audio(self, remote_audio_track):
        stream = rtc.AudioStream(remote_audio_track)
        self.logger.info('{"step":"agent.audio_stream","status":"opened"}')
        try:
            async for ev in stream:
                data = getattr(ev, "data", None)
                if ev is None or data is None:
                    continue
                if hasattr(data, "tobytes"):
                    data = data.tobytes()
                elif not isinstance(data, (bytes, bytearray)):
                    data = bytes(data)
                num_channels = getattr(ev, "num_channels", 1)
                pcm16k = resample_to_16k_mono(data, num_channels, getattr(ev, "sample_rate", 48000))
                energy = self._rms16(pcm16k)
                async with self.buffer_lock:
                    self.buffer.extend(pcm16k)
                self.last_audio_ts_ms = int(time.time() * 1000)
                if (self.last_audio_ts_ms // 500) % 2 == 0:
                    self.logger.info('{"step":"agent.audio_frame","energy":%.2f,"buffer":%d}', energy, len(self.buffer))
        except Exception as exc:
            self.report_error("consume-audio", exc)
        finally:
            await stream.aclose()
            self.logger.info('{"step":"agent.audio_stream","status":"closed"}')

    async def _ensure_subscription(self, publication):
        try:
            result = publication.set_subscribed(True)
            if asyncio.iscoroutine(result):
                await result
            log_step("agent.track_subscribe_request", "success", {"track_sid": getattr(publication, "sid", None)})
        except Exception as exc:
            log_step("agent.track_subscribe_request", "failed", {"error": repr(exc)})

    async def _watchdog_loop(self):
        self.logger.info('{"step":"agent.watchdog","status":"started"}')
        try:
            while True:
                await asyncio.sleep(0.2)
                now_ms = int(time.time() * 1000)
                async with self.buffer_lock:
                    buf_len = len(self.buffer)
                idle_ms = now_ms - self.last_audio_ts_ms
                if buf_len > 0 and idle_ms >= self.idle_flush_ms:
                    self.logger.info(
                        '{"step":"agent.watchdog","action":"flush","idle_ms":%d,"buffer":%d}', idle_ms, buf_len
                    )
                    await self.on_final_chunk()
        except asyncio.CancelledError:
            self.logger.info('{"step":"agent.watchdog","status":"stopped"}')
        except Exception as exc:
            self.report_error("watchdog", exc)

    async def on_final_chunk(self):
        async with self.buffer_lock:
            if not self.buffer:
                return
            pcm = bytes(self.buffer)
            self.buffer.clear()
            self.vad.reset()

        chunk_len = len(pcm)
        self.logger.info("Final chunk: %d bytes", chunk_len)
        log_step("agent.pipeline", "chunk_ready", {"bytes": chunk_len})

        try:
            wav_bytes = self._pcm16_to_wav(pcm, sample_rate=16_000)
            self.logger.info('{"step":"agent.pipeline","stage":"stt","bytes":%d}', len(wav_bytes))
            log_step("sarvam.stt", "request", {"bytes": len(wav_bytes)})
            self.diag["stage"] = "stt"
            transcript = await self.sarvam_stt(wav_bytes)
            self.logger.info('{"step":"agent.pipeline","stage":"stt.done","text":"%s"}', transcript)
            if not transcript:
                self.logger.info("Empty transcript.")
                log_step("sarvam.stt", "empty", None)
                self.diag["stage"] = "idle"
                return

            self.logger.info("Transcript received: %s", transcript)
            log_step("sarvam.stt", "response", {"text": transcript})
            self.last_transcript = transcript
            self.diag["last_transcript"] = transcript
            self.diag["stage"] = "rag"
            await self.send_text(f"[user] {transcript}")

            context = self.rag.retrieve_context(transcript)
            context_len = len(context or "")
            self.logger.debug("RAG context chars: %d", context_len)
            log_step("rag.lookup", "complete", {"chars": context_len})
            self.logger.info('{"step":"agent.pipeline","stage":"chat"}')
            reply_text = await self.sarvam_chat(transcript, context=context)
            self.logger.info('{"step":"agent.pipeline","stage":"chat.done","chars":%d}', len(reply_text))
            self.logger.info("Assistant reply text chars: %d", len(reply_text))
            log_step("sarvam.chat", "response", {"chars": len(reply_text)})
            self.diag["stage"] = "chat"
            self.diag["last_reply_chars"] = len(reply_text)

            await self.stream_text(reply_text)

            self.logger.info('{"step":"agent.pipeline","stage":"tts"}')
            tts_wav = await self.sarvam_tts(reply_text)
            self.logger.info('{"step":"agent.pipeline","stage":"tts.done","bytes":%d}', len(tts_wav))
            self.logger.info("Sarvam TTS returned %d bytes", len(tts_wav))
            log_step("sarvam.tts", "response", {"bytes": len(tts_wav)})
            self.diag["stage"] = "tts"
            self.diag["last_tts_bytes"] = len(tts_wav)
            await self.stream_bytes(tts_wav)

        except Exception as exc:
            self.report_error("pipeline", exc)
            self.diag["stage"] = "error"
        finally:
            self.vad.reset()
            if self.diag.get("stage") != "error":
                self.diag["stage"] = "idle"

    async def send_text(self, text: str):
        await self._publish_bytes(text.encode("utf-8"), reliable=True, topic="assistant-text")

    async def stream_text(self, text: str):
        await self._publish_chunked(text.encode("utf-8"), reliable=True, topic="assistant-text")

    async def stream_bytes(self, data: bytes):
        await self._publish_chunked(data, reliable=False, topic="assistant-tts.wav")

    async def _publish_chunked(self, data: bytes, *, reliable: bool, topic: str):
        for i in range(0, len(data), self.MAX_DATA_CHUNK):
            await self._publish_bytes(data[i : i + self.MAX_DATA_CHUNK], reliable=reliable, topic=topic)
            await asyncio.sleep(0)

    async def _publish_bytes(self, payload: bytes, *, reliable: bool, topic: str):
        state = getattr(self.room, "connection_state", None) if self.room else None
        is_connected, _, _ = _normalize_conn_state(state)
        if not self.room or not is_connected:
            self.logger.warning("Room not connected; skipping publish (%s, %d bytes).", topic, len(payload))
            log_step("agent.publish", "skipped", {"topic": topic, "bytes": len(payload)})
            return
        try:
            self.logger.info(
                '{"step":"agent.publish","topic":"%s","bytes":%d,"reliable":%s}',
                topic,
                len(payload),
                str(reliable).lower(),
            )
            await self.room.local_participant.publish_data(payload, reliable=reliable, topic=topic)
            self.logger.info('{"step":"agent.publish","topic":"%s","status":"ok"}', topic)
            log_step("agent.publish", "sent", {"topic": topic, "bytes": len(payload)})
        except Exception as exc:
            self.report_error(f"publish[{topic}]", exc)
            raise

    def _rms16(self, pcm_bytes: bytes) -> float:
        import array
        import math

        if not pcm_bytes:
            return 0.0
        a = array.array("h")
        a.frombytes(pcm_bytes[: (len(pcm_bytes) // 2) * 2])
        if not a:
            return 0.0
        s = sum(int(x) * int(x) for x in a)
        return math.sqrt(s / len(a))

    async def sarvam_stt(self, wav_bytes: bytes) -> str:
        if not self.settings.sarvam_api_key:
            log_step("sarvam.stt", "demo_mode", None)
            return ""
        if self.sarvam_sync:
            try:
                buf = io.BytesIO(wav_bytes)
                buf.name = "audio.wav"
                resp = self.sarvam_sync.speech_to_text.transcribe(
                    file=buf,
                    model="saarika:v2.5",
                    language_code=self.settings.sarvam_language_code,
                )
                text = (resp or {}).get("text", "")
                if text:
                    return text
            except Exception as exc:
                self.report_error("sarvam-stt-sdk", exc)
        if self.settings.sarvam_stt_transport == "ws":
            return await self._sarvam_stt_ws(wav_bytes)
        return await self._sarvam_stt_rest(wav_bytes)

    async def sarvam_chat(self, user_text: str, context: Optional[str] = None) -> str:
        if not self.settings.sarvam_api_key:
            log_step("sarvam.chat", "demo_mode", None)
            return f"(demo) You said: {user_text}"
        url = "https://api.sarvam.ai/v1/chat"
        headers = {"Authorization": f"Bearer {self.settings.sarvam_api_key}", "Content-Type": "application/json"}
        body = {
            "messages": [
                {"role": "system", "content": "You are a concise helpful assistant."},
                {"role": "user", "content": user_text},
                {"role": "system", "content": f"Context:\n{context or ''}"},
            ]
        }
        log_step("sarvam.chat.http", "request", {"user_chars": len(user_text), "context_chars": len(context or "")})
        try:
            resp = await self.http.post(url, headers=headers, json=body)
            resp.raise_for_status()
            data = resp.json()
            reply = data.get("reply") or data.get("choices", [{}])[0].get("message", {}).get("content", "")
            log_step("sarvam.chat.http", "response", {"status": resp.status_code, "reply_preview": reply[:120]})
            return reply
        except Exception as exc:
            self.report_error("sarvam-chat", exc)
            return "(error generating reply)"

    async def sarvam_tts(self, text: str) -> bytes:
        if not self.settings.sarvam_api_key:
            log_step("sarvam.tts", "demo_mode", None)
            return b""
        if self.sarvam_async:
            try:
                chunks: list[bytes] = []
                speaker = (
                    self.settings.sarvam_tts_voice.split(":")[0]
                    if ":" in self.settings.sarvam_tts_voice
                    else self.settings.sarvam_tts_voice
                )
                async for event in self.sarvam_async.audio.tts.stream(
                    text=text,
                    model="bulbul:v2",
                    target_language_code=self.settings.sarvam_language_code,
                    speaker=speaker,
                    audio_format="wav",
                ):
                    audio = getattr(event, "audio", None)
                    if audio:
                        chunks.append(audio)
                if chunks:
                    return b"".join(chunks)
            except Exception as exc:
                self.report_error("sarvam-tts-sdk", exc)
        if self.settings.sarvam_tts_transport == "ws":
            return await self._sarvam_tts_ws(text)
        return await self._sarvam_tts_rest(text)

    async def _sarvam_stt_rest(self, wav_bytes: bytes) -> str:
        url = "https://api.sarvam.ai/v1/stt"
        headers = {"Authorization": f"Bearer {self.settings.sarvam_api_key}"}
        files = {"file": ("audio.wav", wav_bytes, "audio/wav")}
        data = {"language": self.settings.sarvam_language_code}
        log_step("sarvam.stt.rest", "request", {"bytes": len(wav_bytes)})
        try:
            resp = await self.http.post(url, headers=headers, data=data, files=files)
            status = resp.status_code
            body_bytes = resp.content
            if status >= 400:
                self.logger.error(
                    '{"step":"sarvam.stt.rest","status":"error","code":%d,"body":"%s"}',
                    status,
                    body_bytes.decode("utf-8", "ignore"),
                )
                resp.raise_for_status()
            js = resp.json()
            log_step("sarvam.stt.rest", "response", {"status": status, "text": js.get("text", "")})
            return js.get("text", "")
        except Exception as exc:
            self.report_error("sarvam-stt-rest", exc)
            return ""

    async def _sarvam_stt_ws(self, wav_bytes: bytes) -> str:
        ws_url = "wss://api.sarvam.ai/v1/stt/ws"
        headers = [("Authorization", f"Bearer {self.settings.sarvam_api_key}")]
        log_step("sarvam.stt.ws", "request", {"bytes": len(wav_bytes)})
        try:
            async with websockets.connect(ws_url, extra_headers=headers, ping_interval=20, ping_timeout=20) as ws:
                await ws.send(
                    json.dumps(
                        {
                            "language": self.settings.sarvam_language_code,
                            "format": "wav_b64",
                            "audio": base64.b64encode(wav_bytes).decode("ascii"),
                        }
                    )
                )
                result = await ws.recv()
                js = json.loads(result)
                log_step("sarvam.stt.ws", "response", {"raw": js})
                return js.get("text", "")
        except Exception as exc:
            self.report_error("sarvam-stt-ws", exc)
            return ""

    async def _sarvam_tts_rest(self, text: str) -> bytes:
        url = "https://api.sarvam.ai/v1/tts"
        headers = {"Authorization": f"Bearer {self.settings.sarvam_api_key}", "Content-Type": "application/json"}
        body = {
            "text": text,
            "voice": self.settings.sarvam_tts_voice,
            "language": self.settings.sarvam_language_code,
            "format": "wav",
        }
        log_step("sarvam.tts.rest", "request", {"chars": len(text)})
        try:
            resp = await self.http.post(url, headers=headers, json=body)
            status = resp.status_code
            if status >= 400:
                self.logger.error(
                    '{"step":"sarvam.tts.rest","status":"error","code":%d,"body":"%s"}',
                    status,
                    resp.text,
                )
                resp.raise_for_status()
            payload = resp.content
            log_step("sarvam.tts.rest", "response", {"status": status, "bytes": len(payload)})
            return payload
        except Exception as exc:
            self.report_error("sarvam-tts-rest", exc)
            return b""

    async def _sarvam_tts_ws(self, text: str) -> bytes:
        ws_url = "wss://api.sarvam.ai/v1/tts/ws"
        headers = [("Authorization", f"Bearer {self.settings.sarvam_api_key}")]
        log_step("sarvam.tts.ws", "request", {"chars": len(text)})
        try:
            async with websockets.connect(ws_url, extra_headers=headers, ping_interval=20, ping_timeout=20) as ws:
                await ws.send(
                    json.dumps(
                        {
                            "voice": self.settings.sarvam_tts_voice,
                            "language": self.settings.sarvam_language_code,
                            "text": text,
                            "format": "wav_b64",
                        }
                    )
                )
                msg = await ws.recv()
                js = json.loads(msg)
                log_step("sarvam.tts.ws", "response", {"has_audio": "audio_b64" in js})
                if "audio_b64" in js:
                    return base64.b64decode(js["audio_b64"])
                return b""
        except Exception as exc:
            self.report_error("sarvam-tts-ws", exc)
            return b""

    def _pcm16_to_wav(self, pcm: bytes, sample_rate: int = 16_000, num_channels: int = 1) -> bytes:
        import wave

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(num_channels)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm)
        return buf.getvalue()

    def report_error(self, step: str, err: Exception):
        self.logger.error("ERROR[%s]: %s", step, repr(err))
        log_step(step, "error", {"error": repr(err)})
        try:
            msg = f"[error:{step}] {type(err).__name__}: {err}"
            asyncio.create_task(self.send_text(msg))
        except Exception:
            pass

    async def force_flush(self):
        async with self.buffer_lock:
            has_data = len(self.buffer) > 0
        if has_data:
            self.logger.info('{"step":"agent.flush","status":"requested"}')
            await self.on_final_chunk()
            return "requested"
        self.logger.info('{"step":"agent.flush","status":"ignored","reason":"empty-buffer"}')
        return "ignored"

    async def aclose(self):
        if self.watchdog_task:
            self.watchdog_task.cancel()
            try:
                await self.watchdog_task
            except Exception:
                pass
            self.watchdog_task = None
        try:
            await self.http.aclose()
        except Exception:
            pass
        try:
            if self.room:
                await self.room.disconnect()
        except Exception:
            pass

    def diagnostics(self):
        if self.room is not None:
            try:
                is_connected, name_str, val = _normalize_conn_state(self.room.connection_state)
                self.diag.update({"connected": is_connected, "state": name_str, "state_value": val})
            except Exception:
                pass
        if self.last_transcript and not self.diag.get("last_transcript"):
            self.diag["last_transcript"] = self.last_transcript
        return dict(self.diag)
