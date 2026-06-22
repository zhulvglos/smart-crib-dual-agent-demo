"""
run_dual_agent_demo.py

Unified demo script for the Smart Crib Dual-Agent system.

By default, reads existing JSONL event logs and runs the Growth Memory Agent
to generate insights. Use --run-safety to first run the Safety Agent and
produce new event data before analysis.

Usage:
    python run_dual_agent_demo.py                         # Default: analyze existing logs only
    python run_dual_agent_demo.py --run-safety            # Run Safety Agent first, then analyze
    python run_dual_agent_demo.py --serve                 # Analyze and start web server
    python run_dual_agent_demo.py --run-safety --serve    # Full pipeline
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SAFETY_EVENTS_FILE = Path("logs/danger_action_events.jsonl")
SAMPLE_EVENTS_FILE = Path("data/sample_events/danger_action_events.jsonl")
WEB_OUTPUT_FILE = Path("web_demo/data/growth_memory.json")


def find_events_file() -> Path:
    """Find the best available events file."""
    if SAFETY_EVENTS_FILE.exists():
        return SAFETY_EVENTS_FILE
    if SAMPLE_EVENTS_FILE.exists():
        return SAMPLE_EVENTS_FILE
    return SAFETY_EVENTS_FILE


def run_safety_agent():
    """Run the Safety Agent with a mock event to produce new JSONL data."""
    from demo_danger_action import build_danger_action_graph, build_summary, create_mock_danger_event

    print("=" * 72)
    print("[Phase 1] Safety Agent - Running mock danger event pipeline")
    print("=" * 72)
    print()

    event = create_mock_danger_event()
    app = build_danger_action_graph()
    final_state = app.invoke(event)
    summary = build_summary(final_state)

    print("\n  Safety Agent completed.")
    print(f"  Event ID:   {summary['event_id']}")
    print(f"  Risk Level: {summary['risk_level']}")
    print(f"  Log File:   {summary['event_log_file']}")
    print()
    return summary


def run_growth_memory_agent(events_file: Path):
    """Run the Growth Memory Agent on the given events file."""
    from demo_growth_memory_agent import build_growth_memory_graph

    print("=" * 72)
    print("[Phase 2] Growth Memory Agent - Analyzing event logs")
    print("=" * 72)
    print(f"  Events file: {events_file}")
    print()

    app = build_growth_memory_graph()
    final_state = app.invoke({
        "events_file": str(events_file),
        "logs": [],
    })

    output = final_state.get("output", {})
    print()
    print("=" * 72)
    print("Growth Memory Agent Completed")
    print("=" * 72)
    print(f"  Events analyzed: {output.get('event_count', 0)}")
    print(f"  Memory cards:    {len(output.get('memory_cards', []))}")
    print(f"  Suggestions:     {len(output.get('parent_suggestions', []))}")
    print(f"  Safety score:    {output.get('summary_stats', {}).get('safety_score', 'N/A')}")
    print(f"  Output:          {WEB_OUTPUT_FILE}")
    print()

    # Print memory card previews
    cards = output.get("memory_cards", [])
    if cards:
        print("  -- Growth Memory Cards --")
        for i, card in enumerate(cards, 1):
            print(f"  [{i}] {card['title']}")
            print(f"      {card['body'][:80]}...")
        print()

    suggestions = output.get("parent_suggestions", [])
    if suggestions:
        print("  -- Parent Suggestions --")
        for i, s in enumerate(suggestions, 1):
            print(f"  [{i}] {s['title']}: {s['body'][:60]}...")
        print()

    return output


def serve_web():
    """Start the web demo server."""
    import subprocess
    import os

    print("=" * 72)
    print("[Phase 3] Starting Web Demo Server")
    print("=" * 72)

    server_script = Path("web_demo/start_server.py")
    if not server_script.exists():
        print(f"  Warning: {server_script} not found. Please start manually:")
        print(f"    cd web_demo && python start_server.py")
        return

    print("  Starting server at http://localhost:8080")
    print("  Press Ctrl+C to stop.")
    print()

    subprocess.run([sys.executable, str(server_script)], cwd=os.getcwd())


def main():
    parser = argparse.ArgumentParser(description="Smart Crib Dual-Agent Demo")
    parser.add_argument(
        "--run-safety",
        action="store_true",
        help="Run Safety Agent first to produce new event data",
    )
    parser.add_argument(
        "--events-file",
        default=None,
        help="Explicit path to JSONL events file",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Start web demo server after generating data",
    )
    args = parser.parse_args()

    print()
    print("*" * 72)
    print("  Smart Crib Dual-Agent Demo")
    print("  Safety Agent + Growth Memory Agent")
    print("*" * 72)
    print()

    # Phase 1 (optional): Safety Agent
    if args.run_safety:
        run_safety_agent()

    # Determine events file
    if args.events_file:
        events_file = Path(args.events_file)
    else:
        events_file = find_events_file()

    if not events_file.exists():
        print(f"Error: No events file found at {events_file}")
        print("  Run with --run-safety to generate events first,")
        print("  or point to an existing file with --events-file.")
        sys.exit(1)

    # Phase 2: Growth Memory Agent
    run_growth_memory_agent(events_file)

    # Phase 3 (optional): Web server
    if args.serve:
        serve_web()
    else:
        print("To view results in browser:")
        print("  cd web_demo && python start_server.py")
        print("  Then open http://localhost:8080")
        print()


if __name__ == "__main__":
    main()
