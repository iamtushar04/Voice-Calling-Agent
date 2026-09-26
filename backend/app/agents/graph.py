"""
LangGraph Voice Agent — State, Nodes, and Graph.

The voice agent is modeled as an explicit state machine using LangGraph.
Each conversation phase is a graph node; transitions are conditional edges.

States:
  INITIALIZING → GREETING → LISTENING → THINKING → SPEAKING
                                 ↑                      |
                                 └──────────────────────┘
                 TOOL_EXECUTION ↔ THINKING
                 Any state → ENDING → END
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from enum import Enum
from typing import Annotated, Any, Optional, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from app.agents.config import AgentConfig
from app.agents.tools import TOOL_REGISTRY, get_tool_definitions
from app.config import settings
from app.monitoring.logging import get_logger

logger = get_logger(__name__)


# ── Agent State ───────────────────────────────────────────────────────────────

class AgentPhase(str, Enum):
    INITIALIZING = "INITIALIZING"
    GREETING = "GREETING"
    LISTENING = "LISTENING"
    THINKING = "THINKING"
    SPEAKING = "SPEAKING"
    TOOL_EXECUTION = "TOOL_EXECUTION"
    ENDING = "ENDING"
    ENDED = "ENDED"


class VoiceAgentState(TypedDict):
    # Identity
    call_id: str
    contact_id: Optional[str]
    campaign_id: Optional[str]
    agent_config: dict                  # serialized AgentConfig

    # Conversation
    messages: Annotated[list, add_messages]  # full LangChain message history
    current_user_transcript: str        # STT output being built
    current_ai_response: str            # TTS text being built

    # State machine
    phase: str                          # AgentPhase value
    silence_count: int
    barge_in_detected: bool

    # Lifecycle
    ended: bool
    opted_out: bool
    outcome: Optional[str]
    error: Optional[str]

    # Timing (milliseconds, for latency tracking)
    stt_start_ms: Optional[float]
    llm_start_ms: Optional[float]
    tts_start_ms: Optional[float]

    # Callbacks (not serialized — injected at runtime)
    # _audio_sender: callable  ← injected separately, not in TypedDict


def make_initial_state(
    call_id: str,
    agent_config: AgentConfig,
    contact_id: str | None = None,
    campaign_id: str | None = None,
) -> VoiceAgentState:
    return VoiceAgentState(
        call_id=call_id,
        contact_id=contact_id,
        campaign_id=campaign_id,
        agent_config=agent_config.model_dump(),
        messages=[SystemMessage(content=agent_config.system_prompt)],
        current_user_transcript="",
        current_ai_response="",
        phase=AgentPhase.INITIALIZING.value,
        silence_count=0,
        barge_in_detected=False,
        ended=False,
        opted_out=False,
        outcome=None,
        error=None,
        stt_start_ms=None,
        llm_start_ms=None,
        tts_start_ms=None,
    )


# ── LLM Client ────────────────────────────────────────────────────────────────

def _get_llm(agent_config: dict) -> ChatOpenAI:
    model = agent_config.get("llm_model", settings.openai_model)
    tools = get_tool_definitions(agent_config.get("allowed_tools", []))
    llm = ChatOpenAI(
        model=model,
        api_key=settings.openai_api_key,
        streaming=True,
        temperature=0.3,
    )
    if tools:
        llm = llm.bind_tools(tools)
    return llm


# ── Graph Nodes ───────────────────────────────────────────────────────────────

async def initializing_node(state: VoiceAgentState) -> dict:
    """Set up call context. Load contact info if available."""
    logger.info(
        "agent_initializing",
        call_id=state["call_id"],
        contact_id=state["contact_id"],
    )
    return {"phase": AgentPhase.GREETING.value}


async def greeting_node(state: VoiceAgentState) -> dict:
    """
    Generate the initial greeting using LLM.
    The audio output callback (if registered) will stream TTS.
    """
    logger.info("agent_greeting", call_id=state["call_id"])

    agent_config = state["agent_config"]
    llm = _get_llm(agent_config)

    greeting_prompt = agent_config.get(
        "greeting_prompt",
        "Generate a natural, brief phone greeting. Introduce yourself and state the purpose."
    )

    messages = state["messages"] + [HumanMessage(content=greeting_prompt)]
    response = await llm.ainvoke(messages)
    greeting_text = response.content

    logger.info("greeting_generated", text=greeting_text[:100])

    return {
        "messages": [AIMessage(content=greeting_text)],
        "current_ai_response": greeting_text,
        "phase": AgentPhase.SPEAKING.value,
    }


async def listening_node(state: VoiceAgentState) -> dict:
    """
    Wait for user speech input.
    In production, this is driven externally by the WebSocket media stream handler
    which calls graph.update_state() when STT produces a transcript.
    
    Here we return LISTENING phase — the graph pauses here until
    external STT input arrives via interrupt.
    """
    logger.info(
        "agent_listening",
        call_id=state["call_id"],
        silence_count=state["silence_count"],
    )
    return {
        "phase": AgentPhase.LISTENING.value,
        "current_user_transcript": "",
        "stt_start_ms": time.time() * 1000,
    }


async def thinking_node(state: VoiceAgentState) -> dict:
    """
    Send user transcript to LLM, get response.
    Handles both regular responses and tool calls.
    """
    transcript = state["current_user_transcript"]
    logger.info(
        "agent_thinking",
        call_id=state["call_id"],
        user_said=transcript[:100],
    )

    llm_start = time.time() * 1000
    agent_config = state["agent_config"]
    llm = _get_llm(agent_config)

    new_human_msg = HumanMessage(content=transcript)
    all_messages = state["messages"] + [new_human_msg]

    response = await llm.ainvoke(all_messages)

    llm_latency = time.time() * 1000 - llm_start
    logger.info(
        "llm_response_received",
        call_id=state["call_id"],
        latency_ms=round(llm_latency),
        has_tool_calls=bool(response.tool_calls),
    )

    updates: dict = {
        "messages": [new_human_msg, response],
        "llm_start_ms": llm_start,
    }

    if response.tool_calls:
        updates["phase"] = AgentPhase.TOOL_EXECUTION.value
    else:
        updates["current_ai_response"] = response.content
        updates["phase"] = AgentPhase.SPEAKING.value

        # Detect opt-out in response
        if any(
            phrase in transcript.lower()
            for phrase in ["don't call", "stop calling", "remove me", "opt out", "do not call"]
        ):
            updates["opted_out"] = True
            updates["outcome"] = "OPTED_OUT"

    return updates


async def speaking_node(state: VoiceAgentState) -> dict:
    """
    Speak the AI response via TTS.
    In production, this node streams audio back via the WebSocket sender.
    The barge_in_detected flag (set externally) will interrupt this node.
    """
    text = state["current_ai_response"]
    logger.info(
        "agent_speaking",
        call_id=state["call_id"],
        text_preview=text[:80],
        barge_in=state["barge_in_detected"],
    )

    if state["barge_in_detected"]:
        logger.info("barge_in_interrupt", call_id=state["call_id"])
        return {
            "phase": AgentPhase.LISTENING.value,
            "barge_in_detected": False,
            "current_ai_response": "",
        }

    # Check if opted_out → go to ending
    if state["opted_out"]:
        return {"phase": AgentPhase.ENDING.value}

    return {
        "phase": AgentPhase.LISTENING.value,
        "current_ai_response": "",
        "tts_start_ms": time.time() * 1000,
    }


async def tool_execution_node(state: VoiceAgentState) -> dict:
    """
    Execute the LLM's requested tool call.
    Only tools in the allowlist (TOOL_REGISTRY) are permitted.
    """
    last_message = state["messages"][-1]

    if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
        logger.warning("tool_execution_no_tool_calls", call_id=state["call_id"])
        return {"phase": AgentPhase.THINKING.value}

    tool_results = []
    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_call_id = tool_call["id"]

        logger.info(
            "tool_calling",
            call_id=state["call_id"],
            tool_name=tool_name,
            args=tool_args,
        )

        # Allowlist check — critical security control
        if tool_name not in TOOL_REGISTRY:
            result = {"error": f"Tool '{tool_name}' is not permitted."}
            logger.warning(
                "tool_not_allowed",
                call_id=state["call_id"],
                tool_name=tool_name,
            )
        else:
            try:
                tool_fn = TOOL_REGISTRY[tool_name]
                result = await tool_fn(
                    call_id=state["call_id"],
                    contact_id=state["contact_id"],
                    **tool_args,
                )
            except Exception as exc:
                result = {"error": str(exc)}
                logger.error(
                    "tool_execution_error",
                    call_id=state["call_id"],
                    tool_name=tool_name,
                    error=str(exc),
                )

        tool_results.append(
            ToolMessage(
                content=json.dumps(result),
                tool_call_id=tool_call_id,
            )
        )

        # Special handling for opt-out tool
        if tool_name == "mark_opt_out":
            return {
                "messages": tool_results,
                "opted_out": True,
                "outcome": "OPTED_OUT",
                "phase": AgentPhase.THINKING.value,  # LLM will generate goodbye
            }

        # Special handling for end_call tool
        if tool_name == "end_call":
            return {
                "messages": tool_results,
                "phase": AgentPhase.ENDING.value,
            }

    return {
        "messages": tool_results,
        "phase": AgentPhase.THINKING.value,
    }


async def ending_node(state: VoiceAgentState) -> dict:
    """
    Generate a graceful goodbye, then signal call termination.
    The WebSocket handler will detect ended=True and hang up.
    """
    logger.info(
        "agent_ending",
        call_id=state["call_id"],
        opted_out=state["opted_out"],
        outcome=state["outcome"],
    )

    agent_config = state["agent_config"]
    llm = _get_llm(agent_config)

    if state["opted_out"]:
        goodbye_prompt = "The user wants to be removed from our call list. Acknowledge politely and confirm they won't be called again."
    else:
        goodbye_prompt = "Generate a brief, polite goodbye to end the call."

    messages = state["messages"] + [HumanMessage(content=goodbye_prompt)]
    response = await llm.ainvoke(messages)

    return {
        "messages": [response],
        "current_ai_response": response.content,
        "phase": AgentPhase.ENDED.value,
        "ended": True,
    }


# ── Routing Functions ─────────────────────────────────────────────────────────

def route_from_listening(state: VoiceAgentState) -> str:
    """Decide what happens after LISTENING."""
    if state["ended"]:
        return "end"
    if state["current_user_transcript"]:
        return "thinking"
    # Silence handling
    silence_count = state.get("silence_count", 0)
    max_retries = state["agent_config"].get("max_silence_retries", 2)
    if silence_count >= max_retries:
        return "ending"
    return "listening"  # stay in listening, increment silence counter


def route_from_thinking(state: VoiceAgentState) -> str:
    if state["phase"] == AgentPhase.TOOL_EXECUTION.value:
        return "tool_execution"
    if state["phase"] == AgentPhase.ENDING.value:
        return "ending"
    return "speaking"


def route_from_speaking(state: VoiceAgentState) -> str:
    if state["ended"]:
        return "end"
    if state["phase"] == AgentPhase.ENDING.value:
        return "ending"
    return "listening"


# ── Build Graph ───────────────────────────────────────────────────────────────

def build_voice_agent_graph() -> StateGraph:
    """
    Compile the LangGraph voice agent state machine.
    
    Returns a compiled graph that can be invoked or streamed.
    """
    graph = StateGraph(VoiceAgentState)

    # Add all nodes
    graph.add_node("initializing", initializing_node)
    graph.add_node("greeting", greeting_node)
    graph.add_node("listening", listening_node)
    graph.add_node("thinking", thinking_node)
    graph.add_node("speaking", speaking_node)
    graph.add_node("tool_execution", tool_execution_node)
    graph.add_node("ending", ending_node)

    # Entry point
    graph.set_entry_point("initializing")

    # Fixed edges
    graph.add_edge("initializing", "greeting")
    graph.add_edge("greeting", "speaking")
    graph.add_edge("tool_execution", "thinking")
    graph.add_edge("ending", END)

    # Conditional edges
    graph.add_conditional_edges(
        "listening",
        route_from_listening,
        {
            "thinking": "thinking",
            "listening": "listening",
            "ending": "ending",
            "end": END,
        },
    )
    graph.add_conditional_edges(
        "thinking",
        route_from_thinking,
        {
            "speaking": "speaking",
            "tool_execution": "tool_execution",
            "ending": "ending",
        },
    )
    graph.add_conditional_edges(
        "speaking",
        route_from_speaking,
        {
            "listening": "listening",
            "ending": "ending",
            "end": END,
        },
    )

    return graph.compile()


# Compiled graph singleton
voice_agent_graph = build_voice_agent_graph()
