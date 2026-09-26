"""
Database repository layer — all DB access goes through these classes.
No raw SQL or ORM queries should appear in business logic.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import select, update, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models import (
    Agent,
    Call,
    CallEvent,
    CallEventType,
    CallStatus,
    Campaign,
    CampaignContact,
    CampaignContactStatus,
    CampaignStatus,
    Contact,
    ConsentStatus,
)


class AgentRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_by_id(self, agent_id: uuid.UUID) -> Optional[Agent]:
        result = await self.db.execute(select(Agent).where(Agent.id == agent_id))
        return result.scalar_one_or_none()

    async def get_by_name(self, name: str) -> Optional[Agent]:
        result = await self.db.execute(select(Agent).where(Agent.name == name))
        return result.scalar_one_or_none()

    async def create(self, **kwargs) -> Agent:
        agent = Agent(**kwargs)
        self.db.add(agent)
        await self.db.commit()
        await self.db.refresh(agent)
        return agent


class CampaignRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_by_id(self, campaign_id: uuid.UUID) -> Optional[Campaign]:
        result = await self.db.execute(
            select(Campaign).where(Campaign.id == campaign_id)
        )
        return result.scalar_one_or_none()

    async def list_all(self) -> list[Campaign]:
        result = await self.db.execute(select(Campaign).order_by(Campaign.created_at.desc()))
        return list(result.scalars().all())

    async def create(self, **kwargs) -> Campaign:
        campaign = Campaign(**kwargs)
        self.db.add(campaign)
        await self.db.commit()
        await self.db.refresh(campaign)
        return campaign

    async def update_status(
        self,
        campaign_id: uuid.UUID,
        status: CampaignStatus,
        **extra_fields,
    ) -> None:
        values = {"status": status, **extra_fields}
        await self.db.execute(
            update(Campaign).where(Campaign.id == campaign_id).values(**values)
        )
        await self.db.commit()


class ContactRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_by_phone(self, phone_number: str) -> Optional[Contact]:
        result = await self.db.execute(
            select(Contact).where(Contact.phone_number == phone_number)
        )
        return result.scalar_one_or_none()

    async def get_by_id(self, contact_id: uuid.UUID) -> Optional[Contact]:
        result = await self.db.execute(
            select(Contact).where(Contact.id == contact_id)
        )
        return result.scalar_one_or_none()

    async def create(self, **kwargs) -> Contact:
        contact = Contact(**kwargs)
        self.db.add(contact)
        await self.db.commit()
        await self.db.refresh(contact)
        return contact

    async def bulk_create(self, contacts_data: list[dict]) -> list[Contact]:
        contacts = [Contact(**data) for data in contacts_data]
        self.db.add_all(contacts)
        await self.db.commit()
        return contacts

    async def mark_opted_out(self, contact_id: uuid.UUID) -> None:
        await self.db.execute(
            update(Contact)
            .where(Contact.id == contact_id)
            .values(
                do_not_call=True,
                consent_status=ConsentStatus.OPTED_OUT,
                opted_out_at=datetime.now(timezone.utc),
            )
        )
        await self.db.commit()


class CallRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_by_id(self, call_id: uuid.UUID) -> Optional[Call]:
        result = await self.db.execute(select(Call).where(Call.id == call_id))
        return result.scalar_one_or_none()

    async def get_by_provider_id(self, provider_call_id: str) -> Optional[Call]:
        result = await self.db.execute(
            select(Call).where(Call.provider_call_id == provider_call_id)
        )
        return result.scalar_one_or_none()

    async def create(self, **kwargs) -> Call:
        call = Call(**kwargs)
        self.db.add(call)
        await self.db.commit()
        await self.db.refresh(call)
        return call

    async def update(self, call_id: uuid.UUID, **kwargs) -> None:
        await self.db.execute(
            update(Call).where(Call.id == call_id).values(**kwargs)
        )
        await self.db.commit()

    async def get_calls_for_campaign(self, campaign_id: uuid.UUID) -> list[Call]:
        result = await self.db.execute(
            select(Call)
            .where(Call.campaign_id == campaign_id)
            .order_by(Call.started_at.desc())
        )
        return list(result.scalars().all())

    async def add_event(
        self,
        call_id: uuid.UUID,
        event_type: CallEventType,
        payload: dict | None = None,
    ) -> CallEvent:
        event = CallEvent(
            call_id=call_id,
            event_type=event_type,
            payload=payload or {},
        )
        self.db.add(event)
        await self.db.commit()
        return event


class CampaignContactRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_pending(
        self, campaign_id: uuid.UUID, limit: int = 50
    ) -> list[CampaignContact]:
        result = await self.db.execute(
            select(CampaignContact)
            .where(
                and_(
                    CampaignContact.campaign_id == campaign_id,
                    CampaignContact.status == CampaignContactStatus.PENDING,
                )
            )
            .limit(limit)
        )
        return list(result.scalars().all())

    async def update_status(
        self,
        cc_id: uuid.UUID,
        status: CampaignContactStatus,
        **extra,
    ) -> None:
        await self.db.execute(
            update(CampaignContact)
            .where(CampaignContact.id == cc_id)
            .values(status=status, **extra)
        )
        await self.db.commit()

    async def create_bulk(self, records: list[dict]) -> None:
        ccs = [CampaignContact(**r) for r in records]
        self.db.add_all(ccs)
        await self.db.commit()
