# AI Outbound Calling Agent

A production-grade AI voice calling system with real-time streaming conversation, LangGraph state machine, and compliance-first design.

## Quick Start

### 1. Prerequisites

- Python 3.12+
- Docker Desktop (for PostgreSQL + Redis)
- [ngrok](https://ngrok.com) (to expose local webhook to Twilio)
- Twilio account with a phone number
- OpenAI API key
- Deepgram API key
- ElevenLabs API key

### 2. Setup

```bash
# Clone and enter the project
cd voice_agent

# Copy environment file and fill in your credentials
cp .env.example .env
# Edit .env with your actual API keys

# Start PostgreSQL + Redis
docker-compose up postgres redis -d

# Install Python dependencies
cd backend
pip install -r requirements.txt
```

### 3. Start the server

```bash
cd backend
uvicorn app.main:app --reload --port 8000
```

### 4. Expose webhook to Twilio

```bash
# In a separate terminal
ngrok http 8000

# Copy the https URL (e.g. https://abc123.ngrok.io)
# Set it in .env: TWILIO_WEBHOOK_BASE_URL=https://abc123.ngrok.io
```

### 5. Phase 1 — Make a test call

```bash
curl -X POST "http://localhost:8000/voice/test-call?to_number=+1XXXXXXXXXX"
```

Replace `+1XXXXXXXXXX` with your **verified Twilio test number**.

### 6. API Documentation

Open http://localhost:8000/docs for the interactive Swagger UI.

---

## Architecture

```
Phone → Twilio → WebSocket → FastAPI
                                ↓
                         LangGraph Agent
                         ┌──────┴──────┐
                        STT           TTS
                    (Deepgram)   (ElevenLabs)
                         └──────┬──────┘
                              LLM
                           (GPT-4o)
                              ↓
                        PostgreSQL + Redis
```

## Project Structure

```
voice_agent/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI app
│   │   ├── config.py            # Settings from .env
│   │   ├── api/
│   │   │   ├── campaigns.py     # Campaign CRUD + control
│   │   │   ├── calls.py         # Call records + transcripts
│   │   │   └── webhooks.py      # Twilio webhooks + WS media stream
│   │   ├── agents/
│   │   │   ├── graph.py         # LangGraph state machine
│   │   │   ├── config.py        # Agent configuration
│   │   │   └── tools.py         # Tool registry (allowlist)
│   │   ├── telephony/
│   │   │   ├── base.py          # Abstract provider interface
│   │   │   ├── twilio_adapter.py # Twilio implementation
│   │   │   └── media_stream.py  # Real-time audio pipeline
│   │   ├── speech/
│   │   │   ├── stt.py           # Deepgram streaming STT
│   │   │   └── tts.py           # ElevenLabs streaming TTS
│   │   ├── compliance/
│   │   │   └── checker.py       # Pre-call safety checks
│   │   └── database/
│   │       ├── models.py        # SQLAlchemy models
│   │       └── repository.py    # Data access layer
├── .env.example                 # Environment template
├── docker-compose.yml           # Postgres + Redis + Backend
└── sample_contacts.csv          # Test CSV for import
```

## Development Phases

| Phase | Status | Description |
|-------|--------|-------------|
| 1 | ✅ Ready | Single test call + webhook |
| 2 | ✅ Ready | Real-time AI conversation (LangGraph) |
| 3 | ✅ Ready | Database integration |
| 4 | ✅ Ready | CSV contact import |
| 5 | 🔲 Next | Campaign queue + worker |
| 6 | 🔲 Next | Redis concurrency + retry |
| 7 | 🔲 Next | Tool calling integration |
| 8 | 🔲 Next | Post-call summarization |
| 9 | 🔲 Next | Observability + cost tracking |
| 10 | 🔲 Next | Dashboard |

## Compliance

Every call passes through a compliance checker:
- ✅ Valid E.164 phone number
- ✅ Not on DNC list
- ✅ Consent status not OPTED_OUT
- ✅ Campaign is RUNNING
- ✅ Within calling hours

Opt-out requests are recorded immediately and permanently.

## Environment Variables

See [`.env.example`](.env.example) for all required variables.
