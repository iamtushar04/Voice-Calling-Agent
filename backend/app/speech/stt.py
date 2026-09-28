"""
Groq STT client with local Voice Activity Detection (VAD).

Since Groq Whisper API does not support real-time WebSocket streaming (yet),
this module implements a simple local VAD based on audio volume (RMS).
It buffers the audio until silence is detected, then uploads the chunk to Groq.
"""

from __future__ import annotations

import asyncio
import audioop
import io
import time
import wave
from typing import Callable, Optional

import httpx

from app.config import settings
from app.monitoring.logging import get_logger

logger = get_logger(__name__)


class GroqSTTStream:
    """
    Simulates a streaming STT connection using local VAD and Groq API.
    
    Converts incoming μ-law 8kHz audio to PCM.
    Measures volume. Detects speech start and end.
    On speech end, packages audio to WAV and transcribes via Groq.
    """

    def __init__(
        self,
        call_id: str,
        on_transcript: Callable[[str], None],
        on_interim: Optional[Callable[[str], None]] = None,
        on_speech_started: Optional[Callable[[], None]] = None,
        on_silence: Optional[Callable[[], None]] = None,
    ):
        self.call_id = call_id
        self.on_transcript = on_transcript
        self.on_interim = on_interim
        self.on_speech_started = on_speech_started
        self.on_silence = on_silence

        # VAD settings
        self.energy_threshold = 200  # Adjust based on average background noise
        self.silence_timeout = 1.0   # 1 second of silence = end of utterance
        self.min_speech_duration = 0.3 # Ignore sub-300ms blips

        # State
        self._connected = False
        self._is_speaking = False
        self._speech_buffer = bytearray()
        self._last_speech_time = 0.0
        self._speech_start_time = 0.0

        # Background worker for API calls so we don't block audio loop
        self._transcription_queue = asyncio.Queue()
        self._worker_task: Optional[asyncio.Task] = None

    async def connect(self) -> None:
        """Initialize the local VAD stream."""
        logger.debug("stt_connecting", call_id=self.call_id)
        self._connected = True
        self._worker_task = asyncio.create_task(self._transcription_worker())
        logger.info("groq_stt_connected (local VAD active)", call_id=self.call_id)

    async def send_audio(self, audio_bytes_mulaw: bytes) -> None:
        """
        Process inbound audio chunk from Twilio.
        Detect speech, buffer it, trigger transcription on silence.
        """
        if not self._connected or not audio_bytes_mulaw:
            logger.debug("stt_send_audio_skipped", call_id=self.call_id, connected=self._connected, has_audio=bool(audio_bytes_mulaw))
            return

        try:
            # Twilio sends 8kHz μ-law. Convert to 16-bit PCM for energy calculation.
            pcm_bytes = audioop.ulaw2lin(audio_bytes_mulaw, 2)
            
            # Calculate root mean square (volume)
            rms = audioop.rms(pcm_bytes, 2)
            now = time.time()

            if rms > self.energy_threshold:
                # Speech detected!
                self._last_speech_time = now
                if not self._is_speaking:
                    self._is_speaking = True
                    self._speech_start_time = now
                    logger.debug("vad_speech_started", call_id=self.call_id, rms=rms)
                    if self.on_speech_started:
                        await self._call_cb(self.on_speech_started)

            if self._is_speaking:
                # Buffer the audio
                self._speech_buffer.extend(pcm_bytes)

                # Check if silence timeout has been reached
                if now - self._last_speech_time > self.silence_timeout:
                    self._is_speaking = False
                    speech_duration = self._last_speech_time - self._speech_start_time

                    if speech_duration >= self.min_speech_duration:
                        # We have a valid utterance. Queue it for Groq.
                        logger.debug(
                            "vad_utterance_end",
                            call_id=self.call_id,
                            duration=speech_duration,
                        )
                        audio_to_transcribe = bytes(self._speech_buffer)
                        await self._transcription_queue.put(audio_to_transcribe)
                        
                        if self.on_silence:
                            await self._call_cb(self.on_silence)
                    else:
                        logger.debug("vad_ignored_blip", call_id=self.call_id)

                    # Reset buffer
                    self._speech_buffer = bytearray()

        except Exception as exc:
            logger.error("vad_processing_error", call_id=self.call_id, error=str(exc), exc_info=True)

    async def close(self) -> None:
        """Stop VAD and clean up."""
        logger.debug("stt_closing", call_id=self.call_id)
        self._connected = False
        if self._worker_task:
            self._worker_task.cancel()
        logger.info("groq_stt_closed", call_id=self.call_id)

    async def _transcription_worker(self) -> None:
        """Background loop to process buffered audio via Groq."""
        try:
            while True:
                pcm_audio = await self._transcription_queue.get()
                logger.debug("stt_worker_processing", call_id=self.call_id, audio_bytes=len(pcm_audio))
                await self._transcribe_with_groq(pcm_audio)
                self._transcription_queue.task_done()
        except asyncio.CancelledError:
            logger.debug("stt_worker_cancelled", call_id=self.call_id)
            pass

    async def _transcribe_with_groq(self, pcm_audio: bytes) -> None:
        """Package PCM to WAV and send to Groq API."""
        try:
            # 1. Package PCM into an in-memory WAV file
            wav_io = io.BytesIO()
            with wave.open(wav_io, "wb") as wav_file:
                wav_file.setnchannels(1)      # mono
                wav_file.setsampwidth(2)      # 16-bit
                wav_file.setframerate(8000)   # 8kHz
                wav_file.writeframes(pcm_audio)
            
            wav_bytes = wav_io.getvalue()
            logger.debug("stt_wav_created", call_id=self.call_id, wav_bytes=len(wav_bytes))

            # 2. Upload to Groq
            url = "https://api.groq.com/openai/v1/audio/transcriptions"
            headers = {"Authorization": f"Bearer {settings.groq_api_key}"}
            
            files = {
                "file": ("audio.wav", wav_bytes, "audio/wav"),
            }
            data = {
                "model": settings.groq_stt_model,
                "language": "en",
            }

            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(url, headers=headers, files=files, data=data)
                
                if resp.status_code == 200:
                    result = resp.json()
                    transcript = result.get("text", "").strip()
                    if transcript:
                        logger.info(
                            "stt_final_transcript",
                            call_id=self.call_id,
                            transcript=transcript,
                        )
                        if self.on_transcript:
                            await self._call_cb(self.on_transcript, transcript)
                else:
                    logger.error(
                        "groq_stt_api_error",
                        call_id=self.call_id,
                        status_code=resp.status_code,
                        response=resp.text,
                    )

        except Exception as exc:
            logger.error("groq_stt_request_failed", call_id=self.call_id, error=str(exc), exc_info=True)

    @staticmethod
    async def _call_cb(cb, *args):
        """Call callback — supports both sync and async callables."""
        if asyncio.iscoroutinefunction(cb):
            await cb(*args)
        else:
            cb(*args)
