# Sarvam API Quickstart
- STT REST: POST `/speech-to-text` with an audio file (WAV/MP3/PCM...). Returns `transcript`.
- Chat: POST `/v1/chat/completions` with `model: sarvam-m` and `messages`.
- TTS REST: POST `/text-to-speech` with `input_text`, `voice_id` like `bulbul:v2`, returns audio `audio` (base64).
