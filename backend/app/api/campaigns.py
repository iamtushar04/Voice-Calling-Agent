"""
Campaign API — CRUD and lifecycle control.

Endpoints:
  POST   /campaigns               Create campaign
  GET    /campaigns               List campaigns
  GET    /campaigns/{id}          Get campaign details
  POST   /campaigns/{id}/contacts Upload contacts CSV
  POST   /campaigns/{id}/start    Start campaign
  POST   /campaigns/{id}/pause    Pause campaign
  POST   /campaigns/{id}/resume   Resume campaign
  POST   /campaigns/{id}/cancel   Cancel campaign
  GET    /campaigns/{id}/calls    List calls for campaign
"""

from __future__ import annotations

import csv
import io
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.compliance.checker import validate_e164
from app.database.models import (
    CampaignContactStatus,
    CampaignStatus,
    ConsentStatus,
    get_db,
)
from app.database.repository import (
    CampaignContactRepository,
    CampaignRepository,
    CallRepository,
    ContactRepository,
)
from app.monitoring.logging import get_logger

router = APIRouter(prefix="/campaigns", tags=["Campaigns"])
logger = get_logger(__name__)


# ── Request / Response Schemas ─────────────────────────────────────────────────

class CreateCampaignRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    agent_id: str
    max_concurrent_calls: int = Field(default=5, ge=1, le=100)
    timezone: str = "UTC"
    calling_hours_start: str = "09:00"
    calling_hours_end: str = "17:00"


class CampaignResponse(BaseModel):
    id: str
    name: str
    description: Optional[str]
    status: str
    max_concurrent_calls: int
    agent_id: str
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None


class ContactImportResponse(BaseModel):
    total_rows: int
    valid: int
    invalid: int
    duplicates: int
    dnc_skipped: int
    imported: int
    errors: list[str]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("", response_model=CampaignResponse, status_code=201)
async def create_campaign(
    req: CreateCampaignRequest,
    db: AsyncSession = Depends(get_db),
):
    """Create a new campaign in DRAFT status."""
    campaign_repo = CampaignRepository(db)

    try:
        agent_id = uuid.UUID(req.agent_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid agent_id")

    campaign = await campaign_repo.create(
        name=req.name,
        description=req.description,
        agent_id=agent_id,
        max_concurrent_calls=req.max_concurrent_calls,
        timezone=req.timezone,
        calling_hours_start=req.calling_hours_start,
        calling_hours_end=req.calling_hours_end,
        status=CampaignStatus.DRAFT,
    )

    logger.info("campaign_created", campaign_id=str(campaign.id), name=campaign.name)
    return _campaign_to_response(campaign)


@router.get("", response_model=list[CampaignResponse])
async def list_campaigns(db: AsyncSession = Depends(get_db)):
    """List all campaigns ordered by creation date (newest first)."""
    campaign_repo = CampaignRepository(db)
    campaigns = await campaign_repo.list_all()
    return [_campaign_to_response(c) for c in campaigns]


@router.get("/{campaign_id}", response_model=CampaignResponse)
async def get_campaign(campaign_id: str, db: AsyncSession = Depends(get_db)):
    """Get campaign details by ID."""
    campaign = await _get_campaign_or_404(campaign_id, db)
    return _campaign_to_response(campaign)


@router.post("/{campaign_id}/contacts", response_model=ContactImportResponse)
async def import_contacts(
    campaign_id: str,
    file: UploadFile = File(..., description="CSV file with phone_number column"),
    db: AsyncSession = Depends(get_db),
):
    """
    Upload a CSV file of contacts for this campaign.
    
    CSV format:
      phone_number,name,email
      +919876543210,John Doe,john@example.com
    
    Validates each number, normalizes to E.164, checks for duplicates.
    """
    campaign = await _get_campaign_or_404(campaign_id, db)

    if campaign.status not in (CampaignStatus.DRAFT, CampaignStatus.READY):
        raise HTTPException(
            status_code=400,
            detail=f"Cannot add contacts to a {campaign.status} campaign",
        )

    content = await file.read()
    text = content.decode("utf-8-sig")   # handle BOM

    contact_repo = ContactRepository(db)
    cc_repo = CampaignContactRepository(db)

    total_rows = 0
    valid = 0
    invalid = 0
    duplicates = 0
    dnc_skipped = 0
    errors: list[str] = []
    to_create_contacts: list[dict] = []
    seen_in_batch: set[str] = set()

    reader = csv.DictReader(io.StringIO(text))

    for row_num, row in enumerate(reader, start=2):
        total_rows += 1
        raw_phone = (row.get("phone_number") or row.get("phone") or "").strip()

        if not raw_phone:
            invalid += 1
            errors.append(f"Row {row_num}: missing phone_number")
            continue

        # Validate and normalize
        validation = validate_e164(raw_phone)
        if not validation.is_valid:
            invalid += 1
            errors.append(f"Row {row_num}: {validation.error} ('{raw_phone}')")
            continue

        normalized = validation.normalized

        # Deduplication within batch
        if normalized in seen_in_batch:
            duplicates += 1
            continue
        seen_in_batch.add(normalized)

        # Check if already in DB
        existing = await contact_repo.get_by_phone(normalized)
        if existing:
            if existing.do_not_call:
                dnc_skipped += 1
                continue
            duplicates += 1
            # Still link to campaign if not already linked
            contact_id = existing.id
        else:
            # Queue for bulk insert
            to_create_contacts.append({
                "phone_number": normalized,
                "name": (row.get("name") or "").strip() or None,
                "email": (row.get("email") or "").strip() or None,
                "consent_status": ConsentStatus.UNKNOWN,
                "do_not_call": False,
            })
            contact_id = None  # will be set after bulk insert

        valid += 1

    # Bulk insert new contacts
    if to_create_contacts:
        new_contacts = await contact_repo.bulk_create(to_create_contacts)

        # Link all to campaign
        cc_records = [
            {
                "campaign_id": campaign.id,
                "contact_id": c.id,
                "status": CampaignContactStatus.PENDING,
            }
            for c in new_contacts
        ]
        await cc_repo.create_bulk(cc_records)

    # Update campaign status to READY if it was DRAFT
    if campaign.status == CampaignStatus.DRAFT and valid > 0:
        camp_repo = CampaignRepository(db)
        await camp_repo.update_status(campaign.id, CampaignStatus.READY)

    imported = len(to_create_contacts)
    logger.info(
        "contacts_imported",
        campaign_id=campaign_id,
        total=total_rows,
        valid=valid,
        invalid=invalid,
        imported=imported,
    )

    return ContactImportResponse(
        total_rows=total_rows,
        valid=valid,
        invalid=invalid,
        duplicates=duplicates,
        dnc_skipped=dnc_skipped,
        imported=imported,
        errors=errors[:20],   # cap error list
    )


@router.post("/{campaign_id}/start")
async def start_campaign(campaign_id: str, db: AsyncSession = Depends(get_db)):
    """Start a READY or PAUSED campaign."""
    campaign = await _get_campaign_or_404(campaign_id, db)

    if campaign.status not in (CampaignStatus.READY, CampaignStatus.PAUSED):
        raise HTTPException(
            status_code=400,
            detail=f"Campaign must be READY or PAUSED to start (current: {campaign.status})",
        )

    camp_repo = CampaignRepository(db)
    await camp_repo.update_status(
        campaign.id,
        CampaignStatus.RUNNING,
        started_at=datetime.now(timezone.utc),
    )

    logger.info("campaign_started", campaign_id=campaign_id)
    # TODO: Phase 5 — trigger campaign worker to begin dialing
    return {"status": "RUNNING", "campaign_id": campaign_id}


@router.post("/{campaign_id}/pause")
async def pause_campaign(campaign_id: str, db: AsyncSession = Depends(get_db)):
    """Pause a running campaign. In-progress calls continue until completion."""
    campaign = await _get_campaign_or_404(campaign_id, db)

    if campaign.status != CampaignStatus.RUNNING:
        raise HTTPException(status_code=400, detail="Campaign is not RUNNING")

    camp_repo = CampaignRepository(db)
    await camp_repo.update_status(campaign.id, CampaignStatus.PAUSED)

    logger.info("campaign_paused", campaign_id=campaign_id)
    return {"status": "PAUSED", "campaign_id": campaign_id}


@router.post("/{campaign_id}/resume")
async def resume_campaign(campaign_id: str, db: AsyncSession = Depends(get_db)):
    """Resume a paused campaign."""
    campaign = await _get_campaign_or_404(campaign_id, db)

    if campaign.status != CampaignStatus.PAUSED:
        raise HTTPException(status_code=400, detail="Campaign is not PAUSED")

    camp_repo = CampaignRepository(db)
    await camp_repo.update_status(campaign.id, CampaignStatus.RUNNING)

    logger.info("campaign_resumed", campaign_id=campaign_id)
    return {"status": "RUNNING", "campaign_id": campaign_id}


@router.post("/{campaign_id}/cancel")
async def cancel_campaign(campaign_id: str, db: AsyncSession = Depends(get_db)):
    """Cancel a campaign. Cannot be undone."""
    campaign = await _get_campaign_or_404(campaign_id, db)

    if campaign.status == CampaignStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Campaign is already COMPLETED")

    camp_repo = CampaignRepository(db)
    await camp_repo.update_status(
        campaign.id,
        CampaignStatus.CANCELLED,
        completed_at=datetime.now(timezone.utc),
    )

    logger.info("campaign_cancelled", campaign_id=campaign_id)
    return {"status": "CANCELLED", "campaign_id": campaign_id}


@router.get("/{campaign_id}/calls")
async def get_campaign_calls(campaign_id: str, db: AsyncSession = Depends(get_db)):
    """List all calls made in this campaign."""
    await _get_campaign_or_404(campaign_id, db)

    call_repo = CallRepository(db)
    calls = await call_repo.get_calls_for_campaign(uuid.UUID(campaign_id))

    return [
        {
            "call_id": str(c.id),
            "to_number": c.to_number,
            "status": c.status,
            "outcome": c.outcome,
            "duration_seconds": c.duration_seconds,
            "answered_at": c.answered_at.isoformat() if c.answered_at else None,
            "ended_at": c.ended_at.isoformat() if c.ended_at else None,
        }
        for c in calls
    ]


# ── Helpers ───────────────────────────────────────────────────────────────────

async def _get_campaign_or_404(campaign_id: str, db: AsyncSession):
    try:
        cid = uuid.UUID(campaign_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid campaign_id")

    campaign_repo = CampaignRepository(db)
    campaign = await campaign_repo.get_by_id(cid)
    if not campaign:
        raise HTTPException(status_code=404, detail="Campaign not found")
    return campaign


def _campaign_to_response(campaign) -> CampaignResponse:
    return CampaignResponse(
        id=str(campaign.id),
        name=campaign.name,
        description=campaign.description,
        status=campaign.status,
        max_concurrent_calls=campaign.max_concurrent_calls,
        agent_id=str(campaign.agent_id),
        created_at=campaign.created_at.isoformat(),
        started_at=campaign.started_at.isoformat() if campaign.started_at else None,
        completed_at=campaign.completed_at.isoformat() if campaign.completed_at else None,
    )
