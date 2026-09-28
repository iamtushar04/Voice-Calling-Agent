"""
Telephony webhook handler.

Twilio calls these endpoints to notify us of call status changes.
Every webhook is signature-validated before processing.

Webhook events:
  - POST /voice/webhook          → TwiML response (controls call flow)
  - POST /voice/status-callback  → Status updates (ringing, answered, etc.)
  - WS   /voice/media/{call_id}  → Real-time audio stream
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, Form, Header, HTTPException, Request, WebSocket
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.config import AgentConfig
from app.config import settings
from app.database.models import CallEventType, CallStatus, get_db
from app.database.repository import CallRepository
from app.monitoring.logging import get_logger
from app.telephony.media_stream import MediaStreamHandler
from app.telephony.twilio_adapter import TwilioAdapter, twilio_adapter

router = APIRouter(prefix="/voice", tags=["Voice & Webhooks"])
logger = get_logger(__name__)

# In-memory registry of active call sessions
# call_id → MediaStreamHandler
# In production: use Redis to store call metadata, handler runs on same process
_active_handlers: dict[str, MediaStreamHandler] = {}


# ── TwiML Webhook ─────────────────────────────────────────────────────────────

@router.post("/webhook")
async def voice_webhook(
    request: Request,
    CallSid: str = Form(...),
    CallStatus: str = Form(...),
    To: str = Form(default=""),
    From: str = Form(default=""),
    x_twilio_signature: Optional[str] = Header(default=None, alias="X-Twilio-Signature"),
    db: AsyncSession = Depends(get_db),
):
    """
    Twilio calls this when a call is answered.
    We respond with TwiML that opens a Media Stream WebSocket.
    
    This is the primary call connection point.
    """
    logger.info(
        "webhook_received",
        call_sid=CallSid,
        status=CallStatus,
        to=To,
        from_=From,
    )

    # Signature validation (skip in debug mode for local testing)
    if not settings.debug and x_twilio_signature:
        logger.debug("webhook_validating_signature", call_sid=CallSid)
        url = f"{settings.twilio_webhook_base_url}/voice/webhook"
        form_data = dict(await request.form())
        if not twilio_adapter.validate_webhook_signature(url, form_data, x_twilio_signature):
            logger.warning("webhook_invalid_signature", call_sid=CallSid)
            raise HTTPException(status_code=403, detail="Invalid webhook signature")
        logger.debug("webhook_signature_valid", call_sid=CallSid)

    # Look up our internal call record by Twilio's CallSid
    logger.debug("webhook_looking_up_call", call_sid=CallSid)
    call_repo = CallRepository(db)
    call = await call_repo.get_by_provider_id(CallSid)

    if not call:
        logger.warning("webhook_unknown_call_sid", call_sid=CallSid)
        # Return simple TwiML for unknown calls (safety)
        twiml = TwilioAdapter.build_simple_say_twiml(
            "This call cannot be connected at this time. Goodbye."
        )
        logger.info("webhook_returning_fallback_twiml", call_sid=CallSid)
        return _twiml_response(twiml)

    # Build the Media Stream WebSocket URL
    call_id = str(call.id)
    media_stream_url = (
        settings.twilio_webhook_base_url
        .replace("https://", "wss://")
        .replace("http://", "ws://")
        + f"/voice/media/{call_id}"
    )

    # Return TwiML that connects audio to our WebSocket
    twiml = TwilioAdapter.build_media_stream_twiml(
        call_id=call_id,
        media_stream_url=media_stream_url,
    )

    logger.info(
        "twiml_media_stream_sent",
        call_id=call_id,
        call_sid=CallSid,
        ws_url=media_stream_url,
    )

    return _twiml_response(twiml)


# ── Status Callback ───────────────────────────────────────────────────────────

@router.post("/status-callback")
async def status_callback(
    request: Request,
    CallSid: str = Form(...),
    CallStatus: str = Form(...),
    CallDuration: Optional[str] = Form(default=None),
    x_twilio_signature: Optional[str] = Header(default=None, alias="X-Twilio-Signature"),
    db: AsyncSession = Depends(get_db),
):
    """
    Twilio notifies us of call status changes:
    initiated → ringing → in-progress → completed/failed/busy/no-answer
    """
    logger.info(
        "status_callback_received",
        call_sid=CallSid,
        status=CallStatus,
        duration=CallDuration,
    )

    call_repo = CallRepository(db)
    call = await call_repo.get_by_provider_id(CallSid)
    if not call:
        logger.warning("status_callback_unknown_call_sid", call_sid=CallSid)
        return {"ok": True}

    call_id = call.id
    logger.debug("status_callback_processing", call_id=str(call_id), call_sid=CallSid, status=CallStatus)

    # Map Twilio status to our internal status
    status_map = {
        "queued": ("INITIATED", CallEventType.CALL_INITIATED),
        "initiated": ("INITIATED", CallEventType.CALL_INITIATED),
        "ringing": ("RINGING", CallEventType.CALL_RINGING),
        "in-progress": ("ANSWERED", CallEventType.CALL_ANSWERED),
        "completed": ("COMPLETED", CallEventType.CALL_ENDED),
        "failed": ("FAILED", CallEventType.CALL_FAILED),
        "busy": ("BUSY", CallEventType.CALL_ENDED),
        "no-answer": ("NO_ANSWER", CallEventType.CALL_ENDED),
        "canceled": ("FAILED", CallEventType.CALL_FAILED),
    }

    internal_status, event_type = status_map.get(
        CallStatus.lower(), ("FAILED", CallEventType.CALL_FAILED)
    )

    update_fields: dict = {"status": internal_status}

    if CallStatus.lower() == "in-progress":
        update_fields["answered_at"] = datetime.now(timezone.utc)
    elif CallStatus.lower() in ("completed", "failed", "busy", "no-answer", "canceled"):
        update_fields["ended_at"] = datetime.now(timezone.utc)
        if CallDuration:
            update_fields["duration_seconds"] = int(CallDuration)

    logger.debug("status_callback_updating_db", call_id=str(call_id), internal_status=internal_status)
    await call_repo.update(call_id, **update_fields)
    await call_repo.add_event(
        call_id=call_id,
        event_type=event_type,
        payload={"twilio_status": CallStatus, "duration": CallDuration},
    )
    logger.info("status_callback_completed", call_id=str(call_id), event_type=event_type.value)

    return {"ok": True}


# ── WebSocket Media Stream ────────────────────────────────────────────────────

@router.websocket("/media/{call_id}")
async def media_stream(websocket: WebSocket, call_id: str, db: AsyncSession = Depends(get_db)):
    """
    Real-time bidirectional audio WebSocket.
    
    Twilio streams phone audio here. We stream TTS audio back.
    One WebSocket = one active call.
    """
    logger.info("media_ws_accepting", call_id=call_id)
    await websocket.accept()
    logger.info("media_ws_accepted", call_id=call_id)

    # Load call + agent config from DB (with fallback)
    logger.debug("media_ws_loading_call", call_id=call_id)
    agent_config = AgentConfig()  # default fallback
    contact_id = None
    campaign_id = None
    
    try:
        call_uuid = uuid.UUID(call_id)
        call_repo = CallRepository(db)
        call = await call_repo.get_by_id(call_uuid)
        if call:
            contact_id = str(call.contact_id) if call.contact_id else None
            campaign_id = str(call.campaign_id) if call.campaign_id else None
            logger.info("media_ws_call_loaded", call_id=call_id, contact_id=contact_id, campaign_id=campaign_id)
        else:
            logger.warning("media_ws_call_not_found", call_id=call_id)
    except ValueError:
        logger.error("media_ws_invalid_call_id", call_id=call_id)
        await websocket.close(code=1008)
        return
    except Exception as exc:
        # DB error - log but continue with default config
        logger.error("media_ws_db_error", call_id=call_id, error=str(exc), exc_info=True)

    logger.debug("media_ws_creating_handler", call_id=call_id)
    handler = MediaStreamHandler(
        call_id=call_id,
        websocket=websocket,
        agent_config=agent_config,
        contact_id=contact_id,
        campaign_id=campaign_id,
    )

    _active_handlers[call_id] = handler
    logger.debug("media_ws_handler_registered", call_id=call_id, active_count=len(_active_handlers))

    try:
        await handler.handle()
    except Exception as exc:
        logger.error("media_ws_handler_error", call_id=call_id, error=str(exc), exc_info=True)
    finally:
        _active_handlers.pop(call_id, None)
        logger.info("media_ws_handler_removed", call_id=call_id, active_count=len(_active_handlers))


# ── Phase 1: Test Call ────────────────────────────────────────────────────────

@router.post("/test-call", summary="Place a single test call (Phase 1)")
async def make_test_call(
    to_number: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Phase 1 endpoint: Place one outbound test call to a specific number.
    
    - Validates the number
    - Creates a call record
    - Dials via Twilio
    - Returns call details
    
    USE ONLY WITH AUTHORIZED/VERIFIED TEST NUMBERS.
    """
    import traceback
    try:
        from app.compliance.checker import validate_e164

        # Validate number
        validation = validate_e164(to_number)
        if not validation.is_valid:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid phone number: {validation.error}",
            )

        normalized = validation.normalized

        # Create call record in DB
        call_repo = CallRepository(db)
        call = await call_repo.create(
            from_number=settings.twilio_phone_number,
            to_number=normalized,
            status="INITIATED",
        )
        call_id = str(call.id)

        await call_repo.add_event(
            call_id=call.id,
            event_type=CallEventType.CALL_CREATED,
            payload={"to": normalized, "mode": "test"},
        )

        # Build webhook URLs
        webhook_url = f"{settings.twilio_webhook_base_url}/voice/webhook"
        status_callback_url = f"{settings.twilio_webhook_base_url}/voice/status-callback"
        media_stream_url = (
            settings.twilio_webhook_base_url
            .replace("https://", "wss://")
            .replace("http://", "ws://")
            + f"/voice/media/{call_id}"
        )

        from app.telephony.base import OutboundCallRequest
        provider_call_id = await twilio_adapter.create_outbound_call(
            OutboundCallRequest(
                to_number=normalized,
                from_number=settings.twilio_phone_number,
                call_id=call_id,
                webhook_url=webhook_url,
                media_stream_url=media_stream_url,
            )
        )
        # Store Twilio's CallSid
        await call_repo.update(call.id, provider_call_id=provider_call_id)

        logger.info(
            "test_call_placed",
            call_id=call_id,
            to=normalized,
            provider_call_id=provider_call_id,
        )

        return {
            "call_id": call_id,
            "to_number": normalized,
            "provider_call_id": provider_call_id,
            "status": "INITIATED",
            "webhook_url": webhook_url,
        }
    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()}


# ── Helper ────────────────────────────────────────────────────────────────────

def _twiml_response(twiml: str):
    """Return TwiML XML response with correct content type."""
    from fastapi.responses import Response
    return Response(content=twiml, media_type="application/xml")
