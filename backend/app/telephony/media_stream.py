"""
Twilio Media Stream WebSocket handler.

This is the heart of the real-time voice pipeline.

When a call is answered:
  1. Twilio opens a WebSocket connection to ws://your-server/voice/media/{call_id}
  2. This handler receives audio from Twilio (μ-law, 8kHz, base64-encoded)
  3. Audio is forwarded to Deepgram STT in real-time
  4. When STT produces a final transcript, the LangGraph agent is invoked
  5. LLM response is streamed to ElevenLabs TTS
  6. TTS audio is sent back to Twilio (and to the phone)

Barge-in:
  If Deepgram detects speech while TTS is playing:
  → TTS is cancelled
  → LangGraph state is updated with barge_in_detected=True
  → Agent moves back to LISTENING state
"""

from __future__ import annotations

import asyncio
import base64
import json
import time
import uuid
from typing import Optional

from fastapi import WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from app.agents.config import AgentConfig
from app.agents.graph import (
    AgentPhase,
    VoiceAgentState,
    make_initial_state,
    voice_agent_graph,
)
from app.config import settings
from app.monitoring.logging import get_logger
from app.speech.stt import GroqSTTStream
from app.speech.tts import GroqTTSStream

logger = get_logger(__name__)


class MediaStreamHandler:
    """
    Manages the full real-time audio pipeline for one call.
    
    Lifecycle:
      connect → receive audio → STT → LangGraph → TTS → send audio → disconnect
    """

    def __init__(
        self,
        call_id: str,
        websocket: WebSocket,
        agent_config: AgentConfig,
        contact_id: Optional[str] = None,
        campaign_id: Optional[str] = None,
    ):
        self.call_id = call_id
        self.ws = websocket
        self.agent_config = agent_config
        self.contact_id = contact_id
        self.campaign_id = campaign_id

        # Streaming components
        self.stt = GroqSTTStream(
            call_id=call_id,
            on_transcript=self._on_final_transcript,
            on_interim=self._on_interim_transcript,
            on_speech_started=self._on_speech_started,
        )
        self.tts = GroqTTSStream(
            voice_id=agent_config.voice_id,
            call_id=call_id,
        )

        # LangGraph agent state
        self.agent_state: VoiceAgentState = make_initial_state(
            call_id=call_id,
            agent_config=agent_config,
            contact_id=contact_id,
            campaign_id=campaign_id,
        )

        # Stream tracking
        self.stream_sid: Optional[str] = None      # Twilio stream identifier
        self._tts_task: Optional[asyncio.Task] = None
        self._agent_task: Optional[asyncio.Task] = None
        self._is_speaking = False
        self._is_active = True

    # ── Main handler ──────────────────────────────────────────────────────────

    async def handle(self) -> None:
        """Main WebSocket handler — called once per call."""
        logger.info("media_stream_connected", call_id=self.call_id)

        try:
            # Connect to Deepgram
            await self.stt.connect()

            # Start the LangGraph agent (will immediately go to GREETING)
            self._agent_task = asyncio.create_task(self._run_agent_greeting())

            # Main receive loop
            async for raw_message in self.ws.iter_text():
                if not self._is_active:
                    break
                await self._handle_twilio_message(raw_message)

        except WebSocketDisconnect:
            logger.info("media_stream_disconnected", call_id=self.call_id)
        except Exception as exc:
            logger.error(
                "media_stream_error",
                call_id=self.call_id,
                error=str(exc),
                exc_info=True,
            )
        finally:
            await self._cleanup()

    async def _handle_twilio_message(self, raw: str) -> None:
        """Parse and dispatch Twilio Media Stream events."""
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            return

        event = msg.get("event")

        if event == "connected":
            logger.info("twilio_stream_connected", call_id=self.call_id)

        elif event == "start":
            self.stream_sid = msg.get("streamSid")
            logger.info(
                "twilio_stream_started",
                call_id=self.call_id,
                stream_sid=self.stream_sid,
            )

        elif event == "media":
            # Inbound audio from Twilio (user's voice)
            payload = msg.get("media", {}).get("payload", "")
            if payload:
                audio_bytes = base64.b64decode(payload)
                await self.stt.send_audio(audio_bytes)

        elif event == "stop":
            logger.info("twilio_stream_stopped", call_id=self.call_id)
            self._is_active = False

    # ── STT Callbacks ─────────────────────────────────────────────────────────

    async def _on_speech_started(self) -> None:
        """Called by Deepgram VAD when user starts speaking."""
        if self._is_speaking:
            # Barge-in! User interrupted the AI
            logger.info("barge_in_detected", call_id=self.call_id)
            self.tts.cancel()
            if self._tts_task:
                self._tts_task.cancel()
            # Signal the agent graph
            self.agent_state["barge_in_detected"] = True
            self._is_speaking = False

    async def _on_interim_transcript(self, text: str) -> None:
        """Called with partial STT results — used only for barge-in detection."""
        if self._is_speaking and text:
            await self._on_speech_started()

    async def _on_final_transcript(self, text: str) -> None:
        """Called when STT produces a complete user utterance."""
        if not text.strip():
            return

        logger.info(
            "user_transcript_final",
            call_id=self.call_id,
            text=text,
        )

        # Update agent state with transcript, advance graph
        self.agent_state["current_user_transcript"] = text
        self.agent_state["silence_count"] = 0

        # Run LangGraph thinking → speaking cycle
        asyncio.create_task(self._advance_agent())

    # ── LangGraph Agent ───────────────────────────────────────────────────────

    async def _run_agent_greeting(self) -> None:
        """Run the agent through INITIALIZING → GREETING → SPEAKING."""
        try:
            state = await voice_agent_graph.ainvoke(self.agent_state)
            self.agent_state.update(state)

            if state.get("current_ai_response"):
                await self._speak(state["current_ai_response"])

        except Exception as exc:
            logger.error("agent_greeting_error", call_id=self.call_id, error=str(exc))

    async def _advance_agent(self) -> None:
        """Advance the agent graph after receiving user input."""
        try:
            # Update agent with new transcript, run THINKING → SPEAKING
            state = await voice_agent_graph.ainvoke(
                self.agent_state,
                config={"recursion_limit": 10},
            )
            self.agent_state.update(state)

            if state.get("current_ai_response"):
                await self._speak(state["current_ai_response"])

            if state.get("ended"):
                logger.info("agent_ended_call", call_id=self.call_id)
                await asyncio.sleep(1.0)   # brief pause after goodbye
                await self._hangup()

        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.error("agent_advance_error", call_id=self.call_id, error=str(exc))

    # ── TTS & Audio Output ────────────────────────────────────────────────────

    async def _speak(self, text: str) -> None:
        """Stream TTS audio back to Twilio."""
        if not text.strip():
            return

        self.tts.reset()
        self._is_speaking = True

        logger.info(
            "tts_starting",
            call_id=self.call_id,
            text_length=len(text),
        )

        try:
            async for audio_chunk in self.tts.stream(text):
                if not self._is_active:
                    break
                await self._send_audio_to_twilio(audio_chunk)
        except asyncio.CancelledError:
            pass
        finally:
            self._is_speaking = False

    async def _send_audio_to_twilio(self, mulaw_audio: bytes) -> None:
        """
        Send audio chunk back to Twilio via Media Stream WebSocket.
        
        Twilio expects: {"event": "media", "streamSid": "...", "media": {"payload": "<base64>"}}
        """
        if not self.stream_sid or self.ws.client_state != WebSocketState.CONNECTED:
            return

        payload = base64.b64encode(mulaw_audio).decode("utf-8")
        message = json.dumps({
            "event": "media",
            "streamSid": self.stream_sid,
            "media": {"payload": payload},
        })

        try:
            await self.ws.send_text(message)
        except Exception:
            pass

    async def _hangup(self) -> None:
        """Signal Twilio to end the call by closing the WebSocket."""
        logger.info("hanging_up", call_id=self.call_id)
        self._is_active = False
        try:
            if self.ws.client_state == WebSocketState.CONNECTED:
                await self.ws.close()
        except Exception:
            pass

    async def _cleanup(self) -> None:
        """Release all resources."""
        self._is_active = False
        await self.stt.close()
        self.tts.cancel()
        if self._tts_task:
            self._tts_task.cancel()
        if self._agent_task:
            self._agent_task.cancel()
        logger.info("media_stream_cleanup_done", call_id=self.call_id)
