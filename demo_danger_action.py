"""
demo_danger_action.py

Dangerous Action Complete AI Closed-loop Demo.

This module keeps the command-line closed-loop path intentionally lightweight:
mock perception event -> LangGraph decision flow -> simulated TTS warning ->
simulated parent alert -> JSONL event log.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, TypedDict

from langgraph.graph import END, StateGraph


LOG_DIR = Path("logs")
EVENT_LOG_FILE = LOG_DIR / "danger_action_events.jsonl"


class DangerActionState(TypedDict, total=False):
    """State object passed through the dangerous-action closed loop."""

    event_id: str
    state_id: str
    category: str
    source: str
    description: str
    confidence: float
    risk_level: str
    baby_position: str
    detected_at: str
    trigger_mode: str
    video_path: str
    video_progress_when_triggered: float
    frame_index: int
    target_center: Dict[str, int]
    target_bbox: Dict[str, int]
    danger_boundary_x: int
    detector_model: str
    detection_class: str
    detection_confidence: float
    created_at: str

    decision: Dict[str, Any]
    voice_command: Dict[str, Any]
    notification_command: Dict[str, Any]
    event_record: Dict[str, Any]

    logs: List[str]


def now_text() -> str:
    """Return a readable local timestamp."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log_step(state: DangerActionState, message: str) -> DangerActionState:
    """Append and print one closed-loop step log."""
    logs = state.get("logs", [])
    text = f"[{now_text()}] {message}"
    logs.append(text)
    print(text)
    state["logs"] = logs
    return state


def create_mock_danger_event(extra_fields: Dict[str, Any] | None = None) -> DangerActionState:
    """
    Create a simulated dangerous-action perception event.

    In a production product, this input could come from pose detection, object
    detection, radar, or multimodal fusion. In this demo, the event is simulated
    so the AI closed-loop path can be verified end to end.
    """
    created_at = now_text()
    event: DangerActionState = {
        "event_id": f"danger_{int(time.time())}",
        "state_id": "dangerous_action",
        "category": "dangerous_action",
        "source": "mock_vision_sensor",
        "description": "Baby is near the crib edge and appears to lean outward, creating a fall risk.",
        "confidence": 0.93,
        "risk_level": "high",
        "baby_position": "near_crib_edge",
        "detected_at": created_at,
        "created_at": created_at,
        "trigger_mode": "mock_event",
        "logs": [],
    }

    if extra_fields:
        event.update(extra_fields)

    return event


def detect_state_node(state: DangerActionState) -> DangerActionState:
    """Recognize whether the incoming event is a dangerous-action state."""
    log_step(state, "[Perception] Received simulated multimodal safety event.")

    if state.get("state_id") == "dangerous_action":
        log_step(
            state,
            f"[State Recognition] State=dangerous_action, confidence={state.get('confidence')}",
        )
    else:
        log_step(state, "[State Recognition] State is not dangerous_action.")

    return state


def risk_decision_node(state: DangerActionState) -> DangerActionState:
    """Generate the intervention decision based on state, confidence, and position."""
    confidence = state.get("confidence", 0)
    baby_position = state.get("baby_position", "")
    risk_level = state.get("risk_level", "unknown")

    decision = {
        "should_intervene": False,
        "priority": "normal",
        "actions": [],
        "reason": "",
    }

    if (
        state.get("state_id") == "dangerous_action"
        and confidence >= 0.8
        and baby_position == "near_crib_edge"
    ):
        decision = {
            "should_intervene": True,
            "priority": "emergency",
            "actions": [
                "play_voice_warning",
                "send_parent_alert",
                "record_event",
            ],
            "reason": "Baby is near the crib edge with a fall-risk posture; immediate intervention is recommended.",
        }

    state["decision"] = decision

    log_step(
        state,
        (
            "[Agent Decision] "
            f"risk_level={risk_level}, priority={decision['priority']}, "
            f"should_intervene={decision['should_intervene']}"
        ),
    )

    return state


def voice_agent_node(state: DangerActionState) -> DangerActionState:
    """Generate a simulated TTS warning command."""
    decision = state.get("decision", {})

    if not decision.get("should_intervene"):
        log_step(state, "[Voice Agent] No voice intervention required.")
        return state

    voice_command = {
        "type": "tts_warning",
        "state_id": "dangerous_action",
        "text": "Baby, stay safe. Please move back to the middle of the crib.",
        "volume": 0.85,
        "speed": 1.0,
        "repeat": 2,
        "emotion": "firm_but_gentle",
        "simulated": True,
    }

    state["voice_command"] = voice_command

    log_step(
        state,
        (
            "[Voice Agent] Generated simulated TTS command: "
            f"text='{voice_command['text']}', volume={voice_command['volume']}, "
            f"repeat={voice_command['repeat']}"
        ),
    )

    print("\n>>> Simulated TTS Output")
    print(f"    {voice_command['text']}")
    print()

    return state


def notification_agent_node(state: DangerActionState) -> DangerActionState:
    """Generate a simulated parent App/WeChat alert command."""
    decision = state.get("decision", {})

    if not decision.get("should_intervene"):
        log_step(state, "[Notification Agent] No parent alert required.")
        return state

    notification_command = {
        "type": "emergency_alert",
        "channels": ["app_push", "wechat"],
        "state_id": "dangerous_action",
        "title": "High Risk: Baby Near Crib Edge",
        "message": "Detected a dangerous leaning action near the crib edge. Please check immediately.",
        "risk_level": "high",
        "need_parent_action": True,
        "fallback": "If unread for 60 seconds, escalate to SMS or phone call.",
        "simulated": True,
    }

    state["notification_command"] = notification_command

    log_step(
        state,
        f"[Notification Agent] Generated simulated parent alert: {notification_command['title']}",
    )

    print("\n>>> Simulated App/WeChat Push")
    print(f"    Title: {notification_command['title']}")
    print(f"    Message: {notification_command['message']}")
    print(f"    Channels: {', '.join(notification_command['channels'])}")
    print()

    return state


def memory_record_node(state: DangerActionState) -> DangerActionState:
    """Persist the completed event as one JSONL record."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    event_record = {
        "event_id": state.get("event_id"),
        "state_id": state.get("state_id"),
        "category": state.get("category"),
        "source": state.get("source"),
        "description": state.get("description"),
        "confidence": state.get("confidence"),
        "risk_level": state.get("risk_level"),
        "trigger_mode": state.get("trigger_mode"),
        "video_path": state.get("video_path"),
        "video_progress_when_triggered": state.get("video_progress_when_triggered"),
        "frame_index": state.get("frame_index"),
        "target_center": state.get("target_center"),
        "target_bbox": state.get("target_bbox"),
        "danger_boundary_x": state.get("danger_boundary_x"),
        "detector_model": state.get("detector_model"),
        "detection_class": state.get("detection_class"),
        "detection_confidence": state.get("detection_confidence"),
        "baby_position": state.get("baby_position"),
        "detected_at": state.get("detected_at"),
        "created_at": state.get("created_at"),
        "decision": state.get("decision"),
        "voice_command": state.get("voice_command"),
        "notification_command": state.get("notification_command"),
        "closed_loop_finished_at": now_text(),
    }

    state["event_record"] = event_record

    with EVENT_LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event_record, ensure_ascii=False) + "\n")

    log_step(state, f"[Event Log] Written to {EVENT_LOG_FILE}")

    return state


def build_danger_action_graph():
    """Build the dangerous-action closed-loop LangGraph workflow."""
    workflow = StateGraph(DangerActionState)

    workflow.add_node("detect_state", detect_state_node)
    workflow.add_node("risk_decision", risk_decision_node)
    workflow.add_node("voice_agent", voice_agent_node)
    workflow.add_node("notification_agent", notification_agent_node)
    workflow.add_node("memory_record", memory_record_node)

    workflow.set_entry_point("detect_state")

    workflow.add_edge("detect_state", "risk_decision")
    workflow.add_edge("risk_decision", "voice_agent")
    workflow.add_edge("voice_agent", "notification_agent")
    workflow.add_edge("notification_agent", "memory_record")
    workflow.add_edge("memory_record", END)

    return workflow.compile()


def build_summary(final_state: DangerActionState) -> Dict[str, Any]:
    """Build a compact result summary for CLI and video demos."""
    return {
        "event_id": final_state.get("event_id"),
        "state_id": final_state.get("state_id"),
        "risk_level": final_state.get("risk_level"),
        "decision_priority": final_state.get("decision", {}).get("priority"),
        "voice_text": final_state.get("voice_command", {}).get("text"),
        "notification_title": final_state.get("notification_command", {}).get("title"),
        "event_log_file": str(EVENT_LOG_FILE),
    }


def main():
    print("=" * 72)
    print("Dangerous Action Complete AI Closed-loop Demo")
    print("Current trigger mode: mock dangerous_action event")
    print("=" * 72)
    print()

    event = create_mock_danger_event()

    print(">>> Simulated Input Event")
    print(json.dumps(event, ensure_ascii=False, indent=2))
    print()

    app = build_danger_action_graph()

    print(">>> Running AI Closed-loop")
    print()

    final_state = app.invoke(event)

    print()
    print("=" * 72)
    print("Closed-loop Completed")
    print("=" * 72)

    print("\n>>> Final Summary")
    print(json.dumps(build_summary(final_state), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
