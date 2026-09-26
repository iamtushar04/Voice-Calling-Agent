"""
Calls API — retrieve call records, transcripts, and events.

Endpoints:
  GET /calls/{call_id}             Call details
  GET /calls/{call_id}/transcript  Full conversation transcript
  GET /calls/{call_id}/events      Call event timeline
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.database.models import Call, CallEvent, get_db
from app.database.repository import CallRepository
from app.monitoring.logging import get_logger

router = APIRouter(prefix="/calls", tags=["Calls"])
logger = get_logger(__name__)


@router.get("/{call_id}")
async def get_call(call_id: str, db: AsyncSession = Depends(get_db)):
    """Get full call record including summary and outcome."""
    call = await _get_call_or_404(call_id, db)

    return {
        "call_id": str(call.id),
        "campaign_id": str(call.campaign_id) if call.campaign_id else None,
        "contact_id": str(call.contact_id) if call.contact_id else None,
        "provider_call_id": call.provider_call_id,
        "from_number": call.from_number,
        "to_number": call.to_number,
        "status": call.status,
        "outcome": call.outcome,
        "started_at": call.started_at.isoformat() if call.started_at else None,
        "answered_at": call.answered_at.isoformat() if call.answered_at else None,
        "ended_at": call.ended_at.isoformat() if call.ended_at else None,
        "duration_seconds": call.duration_seconds,
        "summary": call.summary,
        "action_items": call.action_items,
        "callback_requested": call.callback_requested,
        "opted_out": call.opted_out,
        "sentiment": call.sentiment,
        "error_code": call.error_code,
        "error_message": call.error_message,
        "cost": {
            "telephony": float(call.telephony_cost or 0),
            "stt": float(call.stt_cost or 0),
            "llm_input": float(call.llm_input_cost or 0),
            "llm_output": float(call.llm_output_cost or 0),
            "tts": float(call.tts_cost or 0),
            "total": float(call.total_cost or 0),
        },
    }


@router.get("/{call_id}/transcript")
async def get_transcript(call_id: str, db: AsyncSession = Depends(get_db)):
    """
    Return the full conversation transcript as an ordered list of messages.
    
    Format:
      [
        {"role": "assistant", "content": "Hello, this is Alex...", "timestamp": "..."},
        {"role": "user",      "content": "Hi, yes I know about it", "timestamp": "..."},
        ...
      ]
    """
    call = await _get_call_or_404(call_id, db)

    return {
        "call_id": call_id,
        "transcript": call.transcript or [],
        "summary": call.summary,
        "outcome": call.outcome,
    }


@router.get("/{call_id}/events")
async def get_call_events(call_id: str, db: AsyncSession = Depends(get_db)):
    """Return the full event timeline for a call."""
    call = await _get_call_or_404(call_id, db)

    result = await db.execute(
        select(CallEvent)
        .where(CallEvent.call_id == call.id)
        .order_by(CallEvent.timestamp)
    )
    events = result.scalars().all()

    return {
        "call_id": call_id,
        "events": [
            {
                "event_type": e.event_type,
                "timestamp": e.timestamp.isoformat(),
                "payload": e.payload,
            }
            for e in events
        ],
    }


async def _get_call_or_404(call_id: str, db: AsyncSession) -> Call:
    try:
        cid = uuid.UUID(call_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid call_id")

    call_repo = CallRepository(db)
    call = await call_repo.get_by_id(cid)
    if not call:
        raise HTTPException(status_code=404, detail="Call not found")
    return call
