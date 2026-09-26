"""
Groq TTS client using Canopy Labs Orpheus models.

Groq's TTS endpoint has a 200-character limit. This stream handles splitting
long text into sentences and querying Groq sequentially.
Returns μ-law 8kHz audio chunks compatible with Twilio Media Streams.
"""

from __future__ import annotations

import audioop
import re
import time
import wave
from typing import AsyncIterator, Optional

import httpx

from app.config import settings
from app.monitoring.logging import get_logger

logger = get_logger(__name__)

GROQ_TTS_URL = "https://api.groq.com/openai/v1/audio/speech"


class GroqTTSStream:
    """
    Streams TTS audio from Groq (Orpheus).
    
    Splits text by sentences to respect the 200 character limit.
    Converts returned WAV to μ-law 8kHz.
    """

    def __init__(
        self,
        voice_id: Optional[str] = None,
        call_id: Optional[str] = None,
    ):
        # Groq doesn't strictly require voice ID for standard, but accepts 'alloy', etc.
        self.voice_id = voice_id or settings.groq_tts_voice or "alloy"
        self.call_id = call_id
        self._cancelled = False

    def cancel(self) -> None:
        """Signal barge-in — stop sending more audio."""
        self._cancelled = True
        logger.info("tts_cancelled", call_id=self.call_id)

    def reset(self) -> None:
        self._cancelled = False

    async def stream(
        self, text: str
    ) -> AsyncIterator[bytes]:
        """
        Stream TTS audio. Yields raw μ-law audio chunks (8kHz, mono) for Twilio.
        """
        if not text.strip():
            return

        self._cancelled = False
        start_time = time.time()
        first_chunk = True

        # Split text into chunks < 200 chars (sentence boundaries preferred)
        text_chunks = self._chunk_text(text, max_len=190)

        headers = {
            "Authorization": f"Bearer {settings.groq_api_key}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            for sentence in text_chunks:
                if self._cancelled:
                    logger.info("tts_stream_cancelled_mid_sentence", call_id=self.call_id)
                    break

                payload = {
                    "model": settings.groq_tts_model,
                    "input": sentence,
                    "voice": self.voice_id,
                    "response_format": "wav",
                }

                resp = await client.post(GROQ_TTS_URL, headers=headers, json=payload)
                
                if resp.status_code != 200:
                    logger.error(
                        "tts_error",
                        call_id=self.call_id,
                        status=resp.status_code,
                        body=resp.text,
                    )
                    continue

                if first_chunk:
                    ttfb = (time.time() - start_time) * 1000
                    logger.info(
                        "tts_first_byte",
                        call_id=self.call_id,
                        ttfb_ms=round(ttfb),
                    )
                    first_chunk = False

                if self._cancelled:
                    break

                # Extract audio from WAV response
                audio_bytes = resp.content
                mulaw_chunk = self._wav_to_mulaw8k(audio_bytes)
                
                if mulaw_chunk:
                    # Yield in small chunks (e.g., 4096 bytes) so Twilio doesn't buffer too much
                    chunk_size = 4096
                    for i in range(0, len(mulaw_chunk), chunk_size):
                        if self._cancelled:
                            break
                        yield mulaw_chunk[i:i+chunk_size]

    def _chunk_text(self, text: str, max_len: int = 190) -> list[str]:
        """Split text by punctuation, ensuring no chunk exceeds max_len."""
        # Simple sentence splitter
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        chunks = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # If a single sentence is too long, split by commas
            if len(sentence) > max_len:
                sub_parts = re.split(r'(?<=,)\s+', sentence)
                for part in sub_parts:
                    if len(part) > max_len:
                        # Hard crop if still too long (rare)
                        part = part[:max_len]
                    if part.strip():
                        chunks.append(part.strip())
            else:
                chunks.append(sentence)
                
        return chunks

    def _wav_to_mulaw8k(self, wav_bytes: bytes) -> bytes:
        """
        Parse WAV header, extract PCM, resample to 8kHz, convert to μ-law.
        Groq Orpheus returns a WAV file (typically 24kHz, 16-bit, mono).
        """
        if not wav_bytes:
            return b""
            
        try:
            import io
            with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
                channels = wav_file.getnchannels()
                sampwidth = wav_file.getsampwidth()
                framerate = wav_file.getframerate()
                
                pcm_data = wav_file.readframes(wav_file.getnframes())
                
                # Convert to mono if stereo
                if channels == 2:
                    pcm_data = audioop.tomono(pcm_data, sampwidth, 1, 1)
                
                # Resample to 8kHz if needed
                if framerate != 8000:
                    pcm_data, _ = audioop.ratecv(pcm_data, sampwidth, 1, framerate, 8000, None)
                
                # Convert PCM (16-bit) to μ-law
                if sampwidth == 2:
                    return audioop.lin2ulaw(pcm_data, 2)
                else:
                    logger.warning("tts_unsupported_sample_width", width=sampwidth)
                    return b""
                    
        except Exception as exc:
            logger.error("tts_audio_conversion_failed", error=str(exc))
            return b""
