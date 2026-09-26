"""
SQLAlchemy async models for the voice agent system.

Tables:
  - agents           : AI agent configurations
  - campaigns        : Calling campaigns
  - contacts         : Phone contacts with consent/DNC status
  - campaign_contacts: Per-contact status within a campaign
  - calls            : Individual call records
  - call_events      : Granular timeline events per call
"""

import enum
import uuid
from datetime import datetime
from decimal import Decimal

from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    DateTime,
    Enum,
    ForeignKey,
    Integer,
    JSON,
    Numeric,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.asyncio import AsyncAttrs, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, relationship

from app.config import settings


# ── Enums ─────────────────────────────────────────────────────────────────────

class CampaignStatus(str, enum.Enum):
    DRAFT = "DRAFT"
    READY = "READY"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"


class ConsentStatus(str, enum.Enum):
    UNKNOWN = "UNKNOWN"
    GRANTED = "GRANTED"
    DENIED = "DENIED"
    OPTED_OUT = "OPTED_OUT"


class CampaignContactStatus(str, enum.Enum):
    PENDING = "PENDING"
    QUEUED = "QUEUED"
    CALLING = "CALLING"
    ANSWERED = "ANSWERED"
    COMPLETED = "COMPLETED"
    NO_ANSWER = "NO_ANSWER"
    BUSY = "BUSY"
    FAILED = "FAILED"
    OPTED_OUT = "OPTED_OUT"
    SKIPPED = "SKIPPED"


class CallStatus(str, enum.Enum):
    INITIATED = "INITIATED"
    RINGING = "RINGING"
    ANSWERED = "ANSWERED"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    NO_ANSWER = "NO_ANSWER"
    BUSY = "BUSY"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class CallOutcome(str, enum.Enum):
    ANSWERED_COMPLETED = "ANSWERED_COMPLETED"
    NO_ANSWER = "NO_ANSWER"
    BUSY = "BUSY"
    VOICEMAIL = "VOICEMAIL"
    FAILED = "FAILED"
    CALLBACK_REQUESTED = "CALLBACK_REQUESTED"
    INTERESTED = "INTERESTED"
    NOT_INTERESTED = "NOT_INTERESTED"
    OPTED_OUT = "OPTED_OUT"
    TRANSFERRED_TO_HUMAN = "TRANSFERRED_TO_HUMAN"
    WRONG_NUMBER = "WRONG_NUMBER"
    UNKNOWN = "UNKNOWN"


class CallEventType(str, enum.Enum):
    CALL_CREATED = "CALL_CREATED"
    CALL_INITIATED = "CALL_INITIATED"
    CALL_RINGING = "CALL_RINGING"
    CALL_ANSWERED = "CALL_ANSWERED"
    AI_STARTED = "AI_STARTED"
    AI_GREETING = "AI_GREETING"
    USER_SPEAKING = "USER_SPEAKING"
    USER_SILENT = "USER_SILENT"
    AI_THINKING = "AI_THINKING"
    AI_SPEAKING = "AI_SPEAKING"
    AI_INTERRUPTED = "AI_INTERRUPTED"
    TOOL_CALLED = "TOOL_CALLED"
    TOOL_RESULT = "TOOL_RESULT"
    OPT_OUT_DETECTED = "OPT_OUT_DETECTED"
    TRANSFER_INITIATED = "TRANSFER_INITIATED"
    CALL_ENDED = "CALL_ENDED"
    CALL_FAILED = "CALL_FAILED"
    POST_CALL_PROCESSED = "POST_CALL_PROCESSED"


# ── Base ──────────────────────────────────────────────────────────────────────

class Base(AsyncAttrs, DeclarativeBase):
    pass


# ── Models ────────────────────────────────────────────────────────────────────

class Agent(Base):
    __tablename__ = "agents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False, unique=True)
    description = Column(Text)
    system_prompt = Column(Text, nullable=False)
    tools_config = Column(JSON, default=list)          # allowed tool names
    llm_model = Column(String(100), default="gpt-4o")
    voice_id = Column(String(255))                     # TTS voice
    settings = Column(JSON, default=dict)              # silence timeout, etc.
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    campaigns = relationship("Campaign", back_populates="agent")


class Campaign(Base):
    __tablename__ = "campaigns"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    status = Column(Enum(CampaignStatus), default=CampaignStatus.DRAFT, nullable=False)
    max_concurrent_calls = Column(Integer, default=5, nullable=False)
    agent_id = Column(UUID(as_uuid=True), ForeignKey("agents.id"), nullable=False)
    timezone = Column(String(50), default="UTC")
    calling_hours_start = Column(String(5), default="09:00")   # HH:MM
    calling_hours_end = Column(String(5), default="17:00")
    metadata_ = Column("metadata", JSON, default=dict)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))

    agent = relationship("Agent", back_populates="campaigns")
    campaign_contacts = relationship("CampaignContact", back_populates="campaign")
    calls = relationship("Call", back_populates="campaign")


class Contact(Base):
    __tablename__ = "contacts"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    phone_number = Column(String(20), nullable=False, unique=True)  # E.164
    name = Column(String(255))
    email = Column(String(255))
    metadata_ = Column("metadata", JSON, default=dict)
    consent_status = Column(Enum(ConsentStatus), default=ConsentStatus.UNKNOWN)
    do_not_call = Column(Boolean, default=False, nullable=False)
    opted_out_at = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    campaign_contacts = relationship("CampaignContact", back_populates="contact")
    calls = relationship("Call", back_populates="contact")


class CampaignContact(Base):
    __tablename__ = "campaign_contacts"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    campaign_id = Column(UUID(as_uuid=True), ForeignKey("campaigns.id"), nullable=False)
    contact_id = Column(UUID(as_uuid=True), ForeignKey("contacts.id"), nullable=False)
    status = Column(
        Enum(CampaignContactStatus),
        default=CampaignContactStatus.PENDING,
        nullable=False,
    )
    attempt_count = Column(Integer, default=0, nullable=False)
    last_attempt_at = Column(DateTime(timezone=True))
    next_attempt_at = Column(DateTime(timezone=True))
    skip_reason = Column(String(255))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    campaign = relationship("Campaign", back_populates="campaign_contacts")
    contact = relationship("Contact", back_populates="campaign_contacts")
    calls = relationship("Call", back_populates="campaign_contact")


class Call(Base):
    __tablename__ = "calls"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    campaign_id = Column(UUID(as_uuid=True), ForeignKey("campaigns.id"))
    contact_id = Column(UUID(as_uuid=True), ForeignKey("contacts.id"))
    campaign_contact_id = Column(UUID(as_uuid=True), ForeignKey("campaign_contacts.id"))
    provider_call_id = Column(String(255), unique=True, index=True)   # Twilio CallSid
    status = Column(Enum(CallStatus), default=CallStatus.INITIATED, nullable=False)
    outcome = Column(Enum(CallOutcome))
    from_number = Column(String(20))     # E.164 caller ID used
    to_number = Column(String(20))       # E.164 dialed number
    started_at = Column(DateTime(timezone=True), server_default=func.now())
    answered_at = Column(DateTime(timezone=True))
    ended_at = Column(DateTime(timezone=True))
    duration_seconds = Column(Integer)
    recording_reference = Column(String(500))
    transcript = Column(JSON, default=list)   # list of {role, content, timestamp}
    summary = Column(Text)
    action_items = Column(JSON, default=list)
    callback_requested = Column(Boolean, default=False)
    callback_time = Column(DateTime(timezone=True))
    opted_out = Column(Boolean, default=False)
    sentiment = Column(String(50))
    error_code = Column(String(100))
    error_message = Column(Text)

    # Cost tracking
    telephony_cost = Column(Numeric(10, 6), default=Decimal("0"))
    stt_cost = Column(Numeric(10, 6), default=Decimal("0"))
    llm_input_cost = Column(Numeric(10, 6), default=Decimal("0"))
    llm_output_cost = Column(Numeric(10, 6), default=Decimal("0"))
    tts_cost = Column(Numeric(10, 6), default=Decimal("0"))
    total_cost = Column(Numeric(10, 6), default=Decimal("0"))

    # Token counts for cost calculation
    llm_input_tokens = Column(Integer, default=0)
    llm_output_tokens = Column(Integer, default=0)
    tts_characters = Column(Integer, default=0)
    stt_seconds = Column(Integer, default=0)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    campaign = relationship("Campaign", back_populates="calls")
    contact = relationship("Contact", back_populates="calls")
    campaign_contact = relationship("CampaignContact", back_populates="calls")
    events = relationship("CallEvent", back_populates="call", order_by="CallEvent.timestamp")


class CallEvent(Base):
    __tablename__ = "call_events"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    call_id = Column(UUID(as_uuid=True), ForeignKey("calls.id"), nullable=False)
    event_type = Column(Enum(CallEventType), nullable=False)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    payload = Column(JSON, default=dict)

    call = relationship("Call", back_populates="events")


# ── Engine & Session ──────────────────────────────────────────────────────────

engine = create_async_engine(
    settings.database_url,
    echo=settings.debug,
)

AsyncSessionLocal = async_sessionmaker(
    engine,
    expire_on_commit=False,
)


async def get_db():
    """FastAPI dependency: yields an async DB session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db():
    """Create all tables. Use Alembic for production migrations."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
