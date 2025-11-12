import io
import numpy as np
from scipy.signal import resample_poly
import wave

TARGET_SR = 16000

def pcm16le_to_float32(pcm_bytes: bytes, num_channels: int, sample_rate: int) -> np.ndarray:
    data = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    if num_channels > 1:
        data = data.reshape(-1, num_channels).mean(axis=1)  # downmix to mono
    return data

def float32_to_pcm16le(x: np.ndarray) -> bytes:
    x = np.clip(x, -1.0, 1.0)
    return (x * 32767.0).astype(np.int16).tobytes()

def resample_to_16k_mono(pcm_bytes: bytes, num_channels: int, sample_rate: int) -> bytes:
    # Resample interleaved int16 PCM to 16k mono int16 bytes.
    x = pcm16le_to_float32(pcm_bytes, num_channels, sample_rate)
    if sample_rate != TARGET_SR:
        x = resample_poly(x, TARGET_SR, sample_rate)
    return float32_to_pcm16le(x)

def wav_bytes_from_pcm16le(pcm_bytes_16k_mono: bytes, sample_rate: int = TARGET_SR) -> bytes:
    # Wrap int16 mono bytes in a WAV container.
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes_16k_mono)
    return buf.getvalue()

class SilenceVAD:
    '''
    Very simple silence-based VAD.
    - Start speech when RMS > start_thresh for >= min_active_ms
    - End speech when RMS < stop_thresh for >= min_silence_ms
    '''
    def __init__(self, start_thresh=0.01, stop_thresh=0.005, min_active_ms=150, min_silence_ms=700, frame_ms=20):
        self.start_thresh = start_thresh
        self.stop_thresh = stop_thresh
        self.min_active_ms = min_active_ms
        self.min_silence_ms = min_silence_ms
        self.frame_ms = frame_ms
        self.reset()

    def reset(self):
        self.in_speech = False
        self.active_ms = 0
        self.silence_ms = 0

    def rms(self, pcm16_mono: bytes) -> float:
        x = np.frombuffer(pcm16_mono, dtype=np.int16).astype(np.float32) / 32768.0
        return float(np.sqrt(np.mean(x * x) + 1e-9))

    def step(self, pcm16_mono: bytes):
        # Return a tuple (event, in_speech) where event is 'start', 'end', or None.
        r = self.rms(pcm16_mono)
        if not self.in_speech:
            if r > self.start_thresh:
                self.active_ms += self.frame_ms
                if self.active_ms >= self.min_active_ms:
                    self.in_speech = True
                    self.silence_ms = 0
                    return "start", True
            else:
                self.active_ms = 0
        else:
            if r < self.stop_thresh:
                self.silence_ms += self.frame_ms
                if self.silence_ms >= self.min_silence_ms:
                    # end of turn
                    self.in_speech = False
                    self.active_ms = 0
                    self.silence_ms = 0
                    return "end", False
            else:
                self.silence_ms = 0
        return None, self.in_speech
