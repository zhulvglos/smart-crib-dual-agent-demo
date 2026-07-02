"""
Voice Companion Agent demo.

Runs a deterministic comfort workflow:
crying event -> comfort memory -> voice strategy -> rule-based script
-> simulated browser playback command -> JSONL comfort result.

No ASR, LLM, voice cloning, or external audio service is used.
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


SAMPLE_CRYING_EVENTS = Path("data/sample_voice/crying_events.jsonl")
VOICE_PREFERENCES = Path("data/sample_voice/voice_preferences.json")
SAMPLE_OUTPUT = Path("data/sample_voice/voice_companion_output.json")
RUNTIME_LOG = Path("logs/voice_companion_events.jsonl")


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def load_crying_event(state: Dict[str, Any]) -> Dict[str, Any]:
    event = deepcopy(state["input_event"])
    required = {
        "event_id",
        "event_type",
        "scene",
        "timestamp",
        "duration_seconds",
        "cry_intensity",
        "baby_id",
    }
    missing = sorted(required - event.keys())
    if missing:
        raise ValueError(f"Missing crying event fields: {', '.join(missing)}")
    if event["event_type"] != "crying":
        raise ValueError("Voice Companion Agent only accepts crying events in this demo.")
    state["crying_event"] = event
    state["trace"].append("load_crying_event")
    return state


def retrieve_comfort_memory(state: Dict[str, Any]) -> Dict[str, Any]:
    event = state["crying_event"]
    preferences = state["preferences"]
    matches = [
        item
        for item in preferences.get("comfort_preferences", [])
        if item.get("scene") == event["scene"]
    ]
    state["matched_memory"] = deepcopy(matches[0]) if matches else None
    state["trace"].append("retrieve_comfort_memory")
    return state


def select_voice_strategy(state: Dict[str, Any]) -> Dict[str, Any]:
    memory = state.get("matched_memory")
    if memory:
        selected_voice = memory["preferred_voice"]
        selected_voice_label = memory["preferred_voice_label"]
        background_audio = memory.get("preferred_background", "none")
        selection_reason = (
            f"命中 {state['crying_event']['scene']} 场景的长期安抚偏好；"
            f"基于 {memory.get('evidence_count', 0)} 次演示样本，"
            f"历史成功率为 {memory.get('historical_success_rate', 0):.0%}。"
        )
    else:
        selected_voice = "default"
        selected_voice_label = "默认音色"
        background_audio = "none"
        selection_reason = "未找到匹配的长期偏好，使用系统默认音色。"

    state["voice_strategy"] = {
        "selected_voice": selected_voice,
        "selected_voice_label": selected_voice_label,
        "background_audio": background_audio,
        "selection_reason": selection_reason,
        "is_simulated": True,
    }
    state["trace"].append("select_voice_strategy")
    return state


def generate_comfort_script(state: Dict[str, Any]) -> Dict[str, Any]:
    event = state["crying_event"]
    selected_voice = state["voice_strategy"]["selected_voice"]

    if event["scene"] == "night_sleep" and selected_voice == "mother":
        script = "宝宝，妈妈在这里。我们慢慢放松，准备睡觉。"
    elif event["scene"] == "daytime_play":
        script = "宝宝别着急，我在陪你。我们一起慢慢安静下来。"
    else:
        script = "宝宝别怕，我在这里陪着你。"

    state["comfort_script"] = script
    state["trace"].append("generate_comfort_script")
    return state


def build_playback_command(state: Dict[str, Any]) -> Dict[str, Any]:
    voice = state["voice_strategy"]["selected_voice"]
    profiles = {
        "mother": {"rate": 0.82, "pitch": 1.12, "volume": 0.82},
        "father": {"rate": 0.86, "pitch": 0.82, "volume": 0.86},
        "default": {"rate": 0.92, "pitch": 1.0, "volume": 0.85},
    }
    state["playback_command"] = {
        "engine": "browser_web_speech_api",
        "voice_role": voice,
        "script": state["comfort_script"],
        "speech": profiles.get(voice, profiles["default"]),
        "background_audio": state["voice_strategy"]["background_audio"],
        "is_simulated": True,
        "fallback": "Use the browser default zh-CN voice when role-specific voices are unavailable.",
    }
    state["trace"].append("build_playback_command")
    return state


def record_comfort_result(
    state: Dict[str, Any],
    *,
    record_runtime: bool,
    runtime_log: Path,
) -> Dict[str, Any]:
    event = state["crying_event"]
    strategy = state["voice_strategy"]
    timestamp = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    record = {
        "event_id": f"voice_{event['event_id']}",
        "source_event_id": event["event_id"],
        "baby_id": event["baby_id"],
        "event_type": "crying_comfort",
        "timestamp": timestamp,
        "scene": event["scene"],
        "selected_voice": strategy["selected_voice"],
        "selected_voice_label": strategy["selected_voice_label"],
        "comfort_script": state["comfort_script"],
        "background_audio": strategy["background_audio"],
        "selection_reason": strategy["selection_reason"],
        "is_simulated": True,
        "outcome": "simulated_calmed_after_3min",
    }
    state["comfort_result"] = record
    state["trace"].append("record_comfort_result")

    if record_runtime:
        runtime_log.parent.mkdir(parents=True, exist_ok=True)
        with runtime_log.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
        state["runtime_log_written"] = str(runtime_log)
    else:
        state["runtime_log_written"] = None
    return state


def run_voice_companion(
    event: Dict[str, Any],
    preferences: Dict[str, Any],
    *,
    record_runtime: bool = True,
    runtime_log: Path = RUNTIME_LOG,
) -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "input_event": event,
        "preferences": preferences,
        "trace": [],
    }
    load_crying_event(state)
    retrieve_comfort_memory(state)
    select_voice_strategy(state)
    generate_comfort_script(state)
    build_playback_command(state)
    record_comfort_result(
        state,
        record_runtime=record_runtime,
        runtime_log=runtime_log,
    )

    return {
        "agent": "Voice Companion Agent",
        "agent_label": "语音陪伴 Agent",
        "workflow_trace": state["trace"],
        "crying_event": state["crying_event"],
        "matched_memory": state["matched_memory"],
        **state["voice_strategy"],
        "comfort_script": state["comfort_script"],
        "playback_command": state["playback_command"],
        "comfort_result": state["comfort_result"],
        "runtime_log_written": state["runtime_log_written"],
        "capability_notice": (
            "当前使用预置角色策略和浏览器 TTS 模拟音色；"
            "真实产品可连接外部语音克隆服务返回的 voice_id。"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Voice Companion Agent demo.")
    parser.add_argument("--events", default=str(SAMPLE_CRYING_EVENTS))
    parser.add_argument("--preferences", default=str(VOICE_PREFERENCES))
    parser.add_argument("--output", default=str(SAMPLE_OUTPUT))
    parser.add_argument("--log", default=str(RUNTIME_LOG))
    args = parser.parse_args()

    events = load_jsonl(Path(args.events))
    if not events:
        raise RuntimeError(f"No crying event found in {args.events}")
    preferences = json.loads(Path(args.preferences).read_text(encoding="utf-8"))

    output = run_voice_companion(
        events[0],
        preferences,
        record_runtime=True,
        runtime_log=Path(args.log),
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=" * 72)
    print("Voice Companion Agent Completed")
    print("=" * 72)
    print(" -> ".join(output["workflow_trace"]))
    print(f"Selected voice: {output['selected_voice_label']}")
    print(f"Comfort script: {output['comfort_script']}")
    print(f"Background: {output['background_audio']}")
    print(f"Runtime log: {output['runtime_log_written']}")
    print(f"Sample output: {output_path}")


if __name__ == "__main__":
    main()
