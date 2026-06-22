"""
generate_growth_memory_web_data.py

Wrapper script to generate Growth Memory Agent web data.
Reads Safety Agent event logs, runs Growth Memory Agent, and outputs JSON
for the web showcase (web_demo/data/growth_memory.json).

Usage:
    python generate_growth_memory_web_data.py
    python generate_growth_memory_web_data.py --events-file data/sample_events/danger_action_events.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from demo_growth_memory_agent import (
    SAMPLE_EVENTS_FILE,
    WEB_OUTPUT_FILE,
    build_growth_memory_graph,
)


def main():
    parser = argparse.ArgumentParser(description="Generate Growth Memory web data")
    parser.add_argument(
        "--events-file",
        default=str(SAMPLE_EVENTS_FILE),
        help="Path to the JSONL events file (default: data/sample_events/danger_action_events.jsonl)",
    )
    args = parser.parse_args()

    events_path = Path(args.events_file)
    if not events_path.exists():
        print(f"Error: Events file not found: {events_path}")
        sys.exit(1)

    print(f"Reading events from: {events_path}")

    app = build_growth_memory_graph()
    final_state = app.invoke({
        "events_file": str(events_path),
        "logs": [],
    })

    print(f"\nDone. Web data written to: {WEB_OUTPUT_FILE}")


if __name__ == "__main__":
    main()
