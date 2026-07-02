"""Generate static web data for the Voice Companion Agent showcase."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from demo_voice_companion_agent import (
    SAMPLE_CRYING_EVENTS,
    SAMPLE_OUTPUT,
    VOICE_PREFERENCES,
    load_jsonl,
    run_voice_companion,
)


WEB_OUTPUT = Path("web_demo/data/voice_companion.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Voice Companion web data.")
    parser.add_argument("--events", default=str(SAMPLE_CRYING_EVENTS))
    parser.add_argument("--preferences", default=str(VOICE_PREFERENCES))
    parser.add_argument("--output", default=str(WEB_OUTPUT))
    args = parser.parse_args()

    events = load_jsonl(Path(args.events))
    preferences = json.loads(Path(args.preferences).read_text(encoding="utf-8"))
    result = run_voice_companion(
        events[0],
        preferences,
        record_runtime=False,
    )

    payload = {
        "generated_at": result["comfort_result"]["timestamp"],
        "demo_mode": "simulated_voice_strategy",
        "is_simulated": True,
        "agent": result["agent"],
        "agent_label": result["agent_label"],
        "workflow_trace": result["workflow_trace"],
        "crying_event": result["crying_event"],
        "matched_memory": result["matched_memory"],
        "selected_voice": result["selected_voice"],
        "selected_voice_label": result["selected_voice_label"],
        "comfort_script": result["comfort_script"],
        "background_audio": result["background_audio"],
        "selection_reason": result["selection_reason"],
        "playback_command": result["playback_command"],
        "simulated_outcome": result["comfort_result"]["outcome"],
        "capability_notice": result["capability_notice"],
        "voice_options": [
            {"id": "mother", "label": "妈妈音色", "description": "柔和、较慢语速"},
            {"id": "father", "label": "爸爸音色", "description": "较低音高、稳定语速"},
            {"id": "default", "label": "默认音色", "description": "浏览器中文默认音色"},
        ],
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    SAMPLE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    SAMPLE_OUTPUT.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Voice Companion web data written to: {output_path}")
    print(f"Sample output written to: {SAMPLE_OUTPUT}")


if __name__ == "__main__":
    main()
