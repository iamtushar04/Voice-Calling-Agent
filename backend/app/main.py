"""
FastAPI application entry point.

Initializes:
  - Logging
  - Database tables
  - All API routers
  - CORS (configured for dev)
  - Health check endpoint
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import campaigns, calls, webhooks
from app.config import settings
from app.database.models import init_db
from app.monitoring.logging import configure_logging, get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle."""
    # ── Startup ───────────────────────────────────────────────────────────────
    configure_logging(debug=settings.debug)
    logger.info(
        "server_starting",
        app_name=settings.app_name,
        version=settings.app_version,
        debug=settings.debug,
    )

    # Create DB tables (use Alembic for production)
    await init_db()
    logger.info("database_initialized")

    yield

    # ── Shutdown ──────────────────────────────────────────────────────────────
    logger.info("server_shutting_down")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="AI Outbound Calling Agent — Backend API",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.debug else ["https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(webhooks.router)
app.include_router(campaigns.router)
app.include_router(calls.router)


# ── Health Check ──────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
async def health_check():
    """Basic health check endpoint."""
    return {
        "status": "ok",
        "app": settings.app_name,
        "version": settings.app_version,
        "timestamp": time.time(),
    }


@app.get("/", tags=["System"])
async def root():
    return {
        "message": f"{settings.app_name} is running",
        "docs": "/docs",
        "health": "/health",
    }
