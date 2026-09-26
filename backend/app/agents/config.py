"""
Agent configuration model.
Prompts, tool allowlists, and settings are configurable — not hardcoded.
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


DEFAULT_SYSTEM_PROMPT = """You are a professional AI phone assistant named Alex.

Your responsibilities:
- Introduce yourself clearly at the start of the call
- Explain the purpose of the call concisely
- Listen carefully to the person's responses
- Never invent information — only use what you know or can look up via tools
- Speak naturally and conversationally (this is a phone call, not a chat)
- Keep responses brief — this is voice, not text
- If someone says they're busy, offer to call back
- If someone says "don't call me again", "remove me", or wants to opt out:
  * Acknowledge politely
  * Call the mark_opt_out tool immediately
  * Confirm they won't be called again
- If someone wants to speak to a human, call transfer_to_human tool
- End the call gracefully when the conversation is complete

Important rules:
- Never make up appointment times, names, or other facts
- Never discuss topics unrelated to the call purpose
- If you don't know something, say so and offer to follow up
- Respect the person's time — be efficient
"""


class AgentConfig(BaseModel):
    """
    Complete configuration for a voice agent instance.
    Stored in the agents table and loaded per call.
    """
    name: str = "Alex"
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    greeting_prompt: str = (
        "Generate a natural, brief opening for a professional phone call. "
        "Introduce yourself as Alex and state you're calling from the company. "
        "Keep it under 2 sentences."
    )
    llm_model: str = "gpt-4o"
    voice_id: str = "21m00Tcm4TlvDq8ikWAM"     # ElevenLabs Rachel
    allowed_tools: list[str] = Field(
        default_factory=lambda: [
            "get_contact_info",
            "mark_opt_out",
            "end_call",
        ]
    )
    # Silence / timing settings
    silence_timeout_seconds: int = 5
    max_silence_retries: int = 2
    initial_greeting_timeout_seconds: int = 10

    # Barge-in sensitivity (0.0 - 1.0)
    vad_sensitivity: float = 0.6

    def model_dump(self, **kwargs) -> dict:  # type: ignore[override]
        return super().model_dump(**kwargs)


# Pre-built agent configs for common use cases
APPOINTMENT_FOLLOWUP_AGENT = AgentConfig(
    name="Appointment Agent",
    system_prompt=DEFAULT_SYSTEM_PROMPT + """

Call purpose: You are following up about an upcoming appointment.
- Confirm the appointment date and time with the customer
- If they need to reschedule, use the reschedule_appointment tool
- If there are any concerns, create a support ticket
""",
    allowed_tools=[
        "get_contact_info",
        "get_appointment",
        "reschedule_appointment",
        "create_ticket",
        "send_followup_sms",
        "mark_opt_out",
        "transfer_to_human",
        "end_call",
    ],
)
