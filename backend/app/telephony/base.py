"""
Telephony provider abstraction layer.

Never couple business logic to Twilio directly.
All telephony interactions go through TelephonyProvider.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Optional


@dataclass
class OutboundCallRequest:
    to_number: str          # E.164
    from_number: str        # E.164 caller ID
    call_id: str            # Our internal call UUID
    webhook_url: str        # POST /voice/webhook
    status_callback_url: Optional[str] = None
    media_stream_url: str = ""   # WS /voice/media/{call_id}


@dataclass
class CallStatusInfo:
    provider_call_id: str
    status: str             # initiated | ringing | answered | completed | failed | busy | no-answer
    duration: Optional[int] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None


class TelephonyProvider(abc.ABC):
    """Abstract base class. Implement a concrete adapter for each provider."""

    @abc.abstractmethod
    async def create_outbound_call(self, request: OutboundCallRequest) -> str:
        """Initiate an outbound call. Returns provider call ID."""
        ...

    @abc.abstractmethod
    async def hangup_call(self, provider_call_id: str) -> None:
        """Immediately terminate a call."""
        ...

    @abc.abstractmethod
    async def get_call_status(self, provider_call_id: str) -> CallStatusInfo:
        """Fetch current call status from the provider."""
        ...

    @abc.abstractmethod
    def validate_webhook_signature(
        self, url: str, params: dict, signature: str
    ) -> bool:
        """Validate that the webhook came from the real provider (not spoofed)."""
        ...
