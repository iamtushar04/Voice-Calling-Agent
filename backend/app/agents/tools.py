"""
Tool registry — ONLY tools listed here can be called by the LLM.

Architecture:
  - LLM requests tool call by name
  - tool_execution_node checks TOOL_REGISTRY allowlist
  - If not in registry → rejected with error (never executed)
  - If allowed → called with (call_id, contact_id, **llm_args)

Each tool is an async callable returning a JSON-serializable dict.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Optional

from langchain_core.tools import tool as lc_tool

from app.monitoring.logging import get_logger

logger = get_logger(__name__)


# ── Tool Implementations ──────────────────────────────────────────────────────

async def tool_get_contact_info(
    call_id: str,
    contact_id: Optional[str] = None,
    **kwargs,
) -> dict:
    """Look up contact information from the database."""
    if not contact_id:
        return {"error": "No contact_id available for this call"}
    # TODO: Replace with real DB lookup via ContactRepository
    return {
        "contact_id": contact_id,
        "name": "Customer",
        "phone_number": "unknown",
        "notes": "No additional notes",
    }


async def tool_get_appointment(
    call_id: str,
    contact_id: Optional[str] = None,
    **kwargs,
) -> dict:
    """Retrieve upcoming appointments for the contact."""
    # TODO: Implement real appointment lookup
    return {
        "appointments": [],
        "message": "No upcoming appointments found",
    }


async def tool_reschedule_appointment(
    call_id: str,
    contact_id: Optional[str] = None,
    appointment_id: Optional[str] = None,
    new_datetime: Optional[str] = None,
    **kwargs,
) -> dict:
    """Reschedule an existing appointment to a new date/time."""
    if not appointment_id or not new_datetime:
        return {"error": "appointment_id and new_datetime are required"}
    # TODO: Implement real rescheduling logic
    return {
        "success": True,
        "appointment_id": appointment_id,
        "new_datetime": new_datetime,
        "message": f"Appointment rescheduled to {new_datetime}",
    }


async def tool_create_ticket(
    call_id: str,
    contact_id: Optional[str] = None,
    subject: str = "Support request",
    description: str = "",
    priority: str = "normal",
    **kwargs,
) -> dict:
    """Create a support ticket for follow-up."""
    # TODO: Integrate with real ticketing system
    ticket_id = f"TICK-{call_id[:8].upper()}"
    return {
        "success": True,
        "ticket_id": ticket_id,
        "subject": subject,
        "message": f"Support ticket {ticket_id} created successfully",
    }


async def tool_send_followup_sms(
    call_id: str,
    contact_id: Optional[str] = None,
    message: str = "",
    **kwargs,
) -> dict:
    """Send a follow-up SMS to the contact after the call."""
    if not message:
        return {"error": "message is required"}
    # TODO: Integrate with Twilio SMS API
    logger.info("sms_queued", call_id=call_id, contact_id=contact_id)
    return {
        "success": True,
        "message": "Follow-up SMS will be sent after the call",
    }


async def tool_mark_opt_out(
    call_id: str,
    contact_id: Optional[str] = None,
    reason: str = "User requested",
    **kwargs,
) -> dict:
    """
    Mark the contact as opted out / do-not-call.
    This is a critical compliance action — must update DB immediately.
    """
    logger.info(
        "opt_out_requested",
        call_id=call_id,
        contact_id=contact_id,
        reason=reason,
    )
    # TODO: Call ContactRepository.mark_opted_out(contact_id) here
    # For now, the state machine handles this via opted_out flag
    return {
        "success": True,
        "message": "Contact marked as opted out. They will not be called again.",
        "contact_id": contact_id,
    }


async def tool_transfer_to_human(
    call_id: str,
    contact_id: Optional[str] = None,
    reason: str = "Customer requested human agent",
    **kwargs,
) -> dict:
    """Initiate transfer of the call to a human agent."""
    logger.info(
        "transfer_to_human_requested",
        call_id=call_id,
        contact_id=contact_id,
        reason=reason,
    )
    # TODO: Integrate with Twilio call transfer or queue system
    return {
        "success": True,
        "message": "Transferring to a human agent now. Please hold.",
        "transfer_initiated": True,
    }


async def tool_end_call(
    call_id: str,
    contact_id: Optional[str] = None,
    reason: str = "Conversation complete",
    **kwargs,
) -> dict:
    """Signal that the call should end gracefully."""
    logger.info("end_call_requested", call_id=call_id, reason=reason)
    return {
        "success": True,
        "message": "Ending call",
        "reason": reason,
    }


# ── Tool Registry (ALLOWLIST) ─────────────────────────────────────────────────
# ONLY tools listed here will ever be executed.
# The LLM cannot call anything outside this dict.

TOOL_REGISTRY: dict[str, Callable] = {
    "get_contact_info":       tool_get_contact_info,
    "get_appointment":        tool_get_appointment,
    "reschedule_appointment": tool_reschedule_appointment,
    "create_ticket":          tool_create_ticket,
    "send_followup_sms":      tool_send_followup_sms,
    "mark_opt_out":           tool_mark_opt_out,
    "transfer_to_human":      tool_transfer_to_human,
    "end_call":               tool_end_call,
}


# ── LangChain Tool Definitions ────────────────────────────────────────────────
# These are the JSON schemas sent to the LLM (OpenAI function calling format)

TOOL_SCHEMAS = {
    "get_contact_info": {
        "name": "get_contact_info",
        "description": "Look up contact information for the person you are speaking with.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    "get_appointment": {
        "name": "get_appointment",
        "description": "Get upcoming appointments for the contact.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    "reschedule_appointment": {
        "name": "reschedule_appointment",
        "description": "Reschedule an existing appointment.",
        "parameters": {
            "type": "object",
            "properties": {
                "appointment_id": {"type": "string", "description": "The appointment ID"},
                "new_datetime": {"type": "string", "description": "New datetime in ISO 8601 format"},
            },
            "required": ["appointment_id", "new_datetime"],
        },
    },
    "create_ticket": {
        "name": "create_ticket",
        "description": "Create a support ticket for follow-up action.",
        "parameters": {
            "type": "object",
            "properties": {
                "subject": {"type": "string"},
                "description": {"type": "string"},
                "priority": {"type": "string", "enum": ["low", "normal", "high"]},
            },
            "required": ["subject"],
        },
    },
    "send_followup_sms": {
        "name": "send_followup_sms",
        "description": "Send a follow-up SMS to the contact after the call.",
        "parameters": {
            "type": "object",
            "properties": {"message": {"type": "string"}},
            "required": ["message"],
        },
    },
    "mark_opt_out": {
        "name": "mark_opt_out",
        "description": "Mark the contact as opted out / do-not-call. Use when the user says they don't want to be called again.",
        "parameters": {
            "type": "object",
            "properties": {"reason": {"type": "string", "description": "Reason for opt-out"}},
            "required": [],
        },
    },
    "transfer_to_human": {
        "name": "transfer_to_human",
        "description": "Transfer the call to a human agent. Use when the customer explicitly requests a human.",
        "parameters": {
            "type": "object",
            "properties": {"reason": {"type": "string"}},
            "required": [],
        },
    },
    "end_call": {
        "name": "end_call",
        "description": "End the call gracefully. Use when the conversation is complete.",
        "parameters": {
            "type": "object",
            "properties": {"reason": {"type": "string"}},
            "required": [],
        },
    },
}


def get_tool_definitions(allowed_tools: list[str]) -> list[dict]:
    """Return tool schemas for tools in the agent's allowlist."""
    return [
        TOOL_SCHEMAS[name]
        for name in allowed_tools
        if name in TOOL_SCHEMAS
    ]
