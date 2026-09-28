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
        logger.info("media_stream_handler_init", call_id=call_id, contact_id=contact_id, campaign_id=campaign_id)
        self.call_id = call_id
        self.ws = websocket
        self.agent_config = agent_config
        self.contact_id = contact_id
        self.campaign_id = campaign_id

        # Streaming components
        logger.debug("media_stream_init_stt", call_id=call_id)
        self.stt = GroqSTTStream(
            call_id=call_id,
            on_transcript=self._on_final_transcript,
            on_interim=self._on_interim_transcript,
            on_speech_started=self._on_speech_started,
        )
        logger.debug("media_stream_init_tts", call_id=call_id, voice_id=agent_config.voice_id)
        self.tts = GroqTTSStream(
            voice_id=agent_config.voice_id,
            call_id=call_id,
        )

        # LangGraph agent state
        logger.debug("media_stream_init_agent_state", call_id=call_id)
        self.agent_state: VoiceAgentState = make_initial_state(
            call_id=call_id,
            agent_config=agent_config,
            contact_id=contact_id,
            campaign_id=campaign_id,
        )

        # Stream tracking
        self.stream_sid: Optional[str] = None  # Twilio stream identifier
        self._stream_ready = asyncio.Event()    
        self._tts_task: Optional[asyncio.Task] = None
        self._agent_task: Optional[asyncio.Task] = None
        self._is_speaking = False
        self._is_active = True
        logger.debug("media_stream_handler_init_complete", call_id=call_id)

    # ── Main handler ──────────────────────────────────────────────────────────

    async def handle(self) -> None:
        """Main WebSocket handler — called once per call."""
        logger.info("media_stream_handle_start", call_id=self.call_id)

        try:
            # Connect to Deepgram
            logger.debug("media_stream_connecting_stt", call_id=self.call_id)
            await self.stt.connect()
            logger.info("media_stream_stt_connected", call_id=self.call_id)

            # Start the LangGraph agent (will immediately go to GREETING)
            logger.debug("media_stream_starting_agent_greeting", call_id=self.call_id)
            self._agent_task = asyncio.create_task(self._run_agent_greeting())

            # Main receive loop
            logger.debug("media_stream_entering_receive_loop", call_id=self.call_id)
            async for raw_message in self.ws.iter_text():
                if not self._is_active:
                    logger.debug("media_stream_inactive_break", call_id=self.call_id)
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
            logger.debug("media_stream_cleanup_start", call_id=self.call_id)
            await self._cleanup()
            logger.info("media_stream_handle_complete", call_id=self.call_id)

    async def _handle_twilio_message(self, raw: str) -> None:
        """Parse and dispatch Twilio Media Stream events."""
        logger.debug("media_stream_received_message", call_id=self.call_id, message_preview=raw[:200])
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("media_stream_invalid_json", call_id=self.call_id, raw=raw[:200])
            return

        event = msg.get("event")
        logger.debug("media_stream_event", call_id=self.call_id, event=event)

        if event == "connected":
            logger.info("twilio_stream_connected", call_id=self.call_id)

        elif event == "start":
            # self.stream_sid = msg.get("streamSid")
            self.stream_sid = msg.get("streamSid") or msg.get("start", {}).get("streamSid")
            self._stream_ready.set()  
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
                logger.debug("media_stream_received_audio", call_id=self.call_id, audio_bytes=len(audio_bytes))
                await self.stt.send_audio(audio_bytes)

        elif event == "stop":
            logger.info("twilio_stream_stopped", call_id=self.call_id)
            self._is_active = False
        else:
            logger.warning("media_stream_unknown_event", call_id=self.call_id, event=event)

    # ── STT Callbacks ─────────────────────────────────────────────────────────

    async def _on_speech_started(self) -> None:
        """Called by Deepgram VAD when user starts speaking."""
        logger.debug("media_stream_speech_started", call_id=self.call_id, is_speaking=self._is_speaking)
        if self._is_speaking:
            # Barge-in! User interrupted the AI
            logger.info("barge_in_detected", call_id=self.call_id)
            self.tts.cancel()
            if self._tts_task:
                self._tts_task.cancel()

            # /Added 
            if self.stream_sid:
                try:
                    await self.ws.send_text(json.dumps({
                    "event": "clear",
                    "streamSid": self.stream_sid,
                    }))
                except Exception:
                    pass
            # Signal the agent graph
            self.agent_state["barge_in_detected"] = True
            self._is_speaking = False

    async def _on_interim_transcript(self, text: str) -> None:
        """Called with partial STT results — used only for barge-in detection."""
        logger.debug("media_stream_interim_transcript", call_id=self.call_id, text=text, is_speaking=self._is_speaking)
        if self._is_speaking and text:
            await self._on_speech_started()

    async def _on_final_transcript(self, text: str) -> None:
        """Called when STT produces a complete user utterance."""
        if not text.strip():
            logger.debug("media_stream_empty_final_transcript", call_id=self.call_id)
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
        logger.debug("media_stream_scheduling_advance_agent", call_id=self.call_id)
        asyncio.create_task(self._advance_agent())

    # ── LangGraph Agent ───────────────────────────────────────────────────────

    async def _run_agent_greeting(self) -> None:
        """Run the agent through INITIALIZING → GREETING → SPEAKING."""
        logger.info("agent_greeting_start", call_id=self.call_id)
        try:
            state = await voice_agent_graph.ainvoke(self.agent_state)
            self.agent_state.update(state)
            logger.debug("agent_greeting_state_updated", call_id=self.call_id, phase=state.get("phase"), has_response=bool(state.get("current_ai_response")))

            if state.get("current_ai_response"):
                await self._speak(state["current_ai_response"])

        except Exception as exc:
            logger.error("agent_greeting_error", call_id=self.call_id, error=str(exc), exc_info=True)

    async def _advance_agent(self) -> None:
        """Advance the agent graph after receiving user input."""
        logger.info("agent_advance_start", call_id=self.call_id, current_phase=self.agent_state.get("phase"))
        try:
            # Update agent with new transcript, run THINKING → SPEAKING
            state = await voice_agent_graph.ainvoke(
                self.agent_state,
                config={"recursion_limit": 10},
            )
            self.agent_state.update(state)
            logger.debug("agent_advance_state_updated", call_id=self.call_id, phase=state.get("phase"), has_response=bool(state.get("current_ai_response")), ended=state.get("ended"))

            if state.get("current_ai_response"):
                await self._speak(state["current_ai_response"])

            if state.get("ended"):
                logger.info("agent_ended_call", call_id=self.call_id)
                await asyncio.sleep(1.0)   # brief pause after goodbye
                await self._hangup()

        except asyncio.CancelledError:
            logger.warning("agent_advance_cancelled", call_id=self.call_id)
            pass
        except Exception as exc:
            logger.error("agent_advance_error", call_id=self.call_id, error=str(exc), exc_info=True)

    # ── TTS & Audio Output ────────────────────────────────────────────────────

    async def _speak(self, text: str) -> None:
        """Stream TTS audio back to Twilio."""
        if not text.strip():
            logger.debug("tts_skipped_empty_text", call_id=self.call_id)
            return
        try:
            logger.debug("tts_waiting_for_stream_ready", call_id=self.call_id)
            await asyncio.wait_for(self._stream_ready.wait(), timeout=15)
        except asyncio.TimeoutError:                           # <-- add
            logger.error("no_stream_sid_timeout", call_id=self.call_id)     # <-- add
            return                                             

        self.tts.reset()
        self._is_speaking = True

        logger.info(
            "tts_starting",
            call_id=self.call_id,
            text_length=len(text),
            text_preview=text[:100],
        )

        try:
            chunk_count = 0
            async for audio_chunk in self.tts.stream(text):
                if not self._is_active:
                    logger.debug("tts_stopped_inactive", call_id=self.call_id)
                    break
                chunk_count += 1
                await self._send_audio_to_twilio(audio_chunk)
            logger.info("tts_completed", call_id=self.call_id, chunks_sent=chunk_count)
        except asyncio.CancelledError:
            logger.warning("tts_cancelled", call_id=self.call_id)
            pass
        finally:
            self._is_speaking = False
            logger.debug("tts_finished", call_id=self.call_id)

    async def _send_audio_to_twilio(self, mulaw_audio: bytes) -> None:
        """
        Send audio chunk back to Twilio via Media Stream WebSocket.
        
        Twilio expects: {"event": "media", "streamSid": "...", "media": {"payload": "<base64>"}}
        """
        if not self.stream_sid or self.ws.client_state != WebSocketState.CONNECTED:
            logger.debug("tts_send_skipped_no_stream_or_disconnected", call_id=self.call_id, has_sid=bool(self.stream_sid), ws_state=self.ws.client_state if self.ws else None)
            return

        payload = base64.b64encode(mulaw_audio).decode("utf-8")
        message = json.dumps({
            "event": "media",
            "streamSid": self.stream_sid,
            "media": {"payload": payload},
        })

        try:
            await self.ws.send_text(message)
            logger.debug("tts_audio_sent", call_id=self.call_id, payload_len=len(payload))
        except Exception as exc:
            logger.warning("tts_audio_send_failed", call_id=self.call_id, error=str(exc))

    async def _hangup(self) -> None:
        """Signal Twilio to end the call by closing the WebSocket."""
        logger.info("hanging_up", call_id=self.call_id)
        self._is_active = False
        try:
            if self.ws.client_state == WebSocketState.CONNECTED:
                await self.ws.close()
                logger.debug("hangup_ws_closed", call_id=self.call_id)
        except Exception as exc:
            logger.warning("hangup_error", call_id=self.call_id, error=str(exc))

    async def _cleanup(self) -> None:
        """Release all resources."""
        logger.debug("media_stream_cleanup_start", call_id=self.call_id)
        self._is_active = False
        await self.stt.close()
        self.tts.cancel()
        if self._tts_task:
            self._tts_task.cancel()
        if self._agent_task:
            self._agent_task.cancel()
        logger.info("media_stream_cleanup_done", call_id=self.call_id)
