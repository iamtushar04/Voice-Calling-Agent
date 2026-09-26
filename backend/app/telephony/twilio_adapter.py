"""
Twilio adapter — concrete implementation of TelephonyProvider.

Uses:
  - Twilio REST API for initiating calls
  - TwiML for call control
  - Twilio Media Streams for real-time audio WebSocket
  - Signature validation for webhook security
"""

from __future__ import annotations

import httpx
from twilio.rest import Client as TwilioClient
from twilio.request_validator import RequestValidator
from twilio.twiml.voice_response import VoiceResponse, Connect, Stream

from app.config import settings
from app.monitoring.logging import get_logger
from app.telephony.base import (
    CallStatusInfo,
    OutboundCallRequest,
    TelephonyProvider,
)

logger = get_logger(__name__)

# Map Twilio call statuses to our internal status strings
TWILIO_STATUS_MAP = {
    "queued": "initiated",
    "initiated": "initiated",
    "ringing": "ringing",
    "in-progress": "answered",
    "completed": "completed",
    "failed": "failed",
    "busy": "busy",
    "no-answer": "no-answer",
    "canceled": "failed",
}


class TwilioAdapter(TelephonyProvider):
    def __init__(self):
        self._client = TwilioClient(
            settings.twilio_account_sid,
            settings.twilio_auth_token,
        )
        self._validator = RequestValidator(settings.twilio_auth_token)
        logger.info("twilio_adapter_initialized")

    async def create_outbound_call(self, request: OutboundCallRequest) -> str:
        """
        Initiate outbound call via Twilio REST API.
        
        When answered, Twilio fetches the webhook_url for TwiML instructions.
        The TwiML will tell Twilio to open a Media Stream WebSocket to our server.
        """
        logger.info(
            "placing_outbound_call",
            to=request.to_number,
            from_=request.from_number,
            call_id=request.call_id,
        )

        # Run sync Twilio SDK in thread pool to avoid blocking the event loop
        import asyncio
        loop = asyncio.get_event_loop()

        call = await loop.run_in_executor(
            None,
            lambda: self._client.calls.create(
                to=request.to_number,
                from_=request.from_number,
                url=request.webhook_url,
            ),
        )

        logger.info(
            "call_created",
            provider_call_id=call.sid,
            call_id=request.call_id,
            status=call.status,
        )
        return call.sid

    async def hangup_call(self, provider_call_id: str) -> None:
        """Terminate an active call."""
        import asyncio
        loop = asyncio.get_event_loop()

        await loop.run_in_executor(
            None,
            lambda: self._client.calls(provider_call_id).update(status="completed"),
        )
        logger.info("call_hung_up", provider_call_id=provider_call_id)

    async def get_call_status(self, provider_call_id: str) -> CallStatusInfo:
        import asyncio
        loop = asyncio.get_event_loop()

        call = await loop.run_in_executor(
            None,
            lambda: self._client.calls(provider_call_id).fetch(),
        )
        return CallStatusInfo(
            provider_call_id=call.sid,
            status=TWILIO_STATUS_MAP.get(call.status, call.status),
            duration=int(call.duration) if call.duration else None,
        )

    def validate_webhook_signature(
        self, url: str, params: dict, signature: str
    ) -> bool:
        """Validate X-Twilio-Signature header to prevent webhook spoofing."""
        return self._validator.validate(url, params, signature)

    # ── TwiML helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def build_media_stream_twiml(call_id: str, media_stream_url: str) -> str:
        """
        Generate TwiML that opens a bidirectional Media Stream WebSocket.
        
        This is returned by the webhook when a call is answered.
        Twilio will connect the phone audio to our WebSocket server.
        """
        response = VoiceResponse()
        connect = Connect()
        stream = Stream(url=media_stream_url)
        stream.parameter(name="call_id", value=call_id)
        connect.append(stream)
        response.append(connect)
        # Keep call alive — the WebSocket controls when to hang up
        response.pause(length=300)
        return str(response)

    @staticmethod
    def build_simple_say_twiml(message: str) -> str:
        """
        Simple TwiML that speaks a message and hangs up.
        Used for Phase 1 test call (no streaming).
        """
        response = VoiceResponse()
        response.say(message, voice="alice")
        response.hangup()
        return str(response)


# Singleton instance
twilio_adapter = TwilioAdapter()
