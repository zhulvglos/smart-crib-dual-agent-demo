"""
demo_danger_action_video.py

PoC Rule-based Video Trigger Demo for the dangerous-action AI closed loop.

Current trigger mode: PoC rule-based video progress trigger.
The video trigger layer is simulated; the backend LangGraph closed loop is
executed for real inside this local demo.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import cv2

from demo_danger_action import (
    EVENT_LOG_FILE,
    build_danger_action_graph,
    build_summary,
    create_mock_danger_event,
)


VIDEO_PATH = Path("data/dangerous_test_video.mp4")
WINDOW_NAME = "AI Crib Safety Demo - PoC Rule-based Trigger"

STAGE_COLORS = {
    "SAFE": (70, 210, 90),
    "WARNING": (0, 190, 255),
    "DANGEROUS_ACTION": (50, 70, 255),
}


def get_stage(progress: float) -> str:
    """Map video progress to a PoC safety stage."""
    if progress < 0.35:
        return "SAFE"
    if progress < 0.70:
        return "WARNING"
    return "DANGEROUS_ACTION"


def draw_text(
    frame,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int] = (245, 245, 245),
    scale: float = 0.58,
    thickness: int = 1,
) -> None:
    """Draw antialiased text with a subtle shadow for screen recording."""
    x, y = origin
    cv2.putText(
        frame,
        text,
        (x + 1, y + 1),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        thickness + 2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_header(frame, stage: str, progress: float, triggered: bool) -> None:
    """Draw the top demo status bar."""
    h, w = frame.shape[:2]
    color = STAGE_COLORS[stage]
    header_h = max(108, int(h * 0.14))

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, header_h), (18, 22, 28), -1)
    cv2.addWeighted(overlay, 0.86, frame, 0.14, 0, frame)
    cv2.line(frame, (0, header_h), (w, header_h), color, 3)

    draw_text(frame, "AI Crib Safety Demo", (24, 34), (255, 255, 255), 0.82, 2)
    draw_text(
        frame,
        "Mode: PoC Rule-based Trigger",
        (24, 68),
        (210, 220, 230),
        0.58,
        1,
    )

    right_x = max(24, int(w * 0.48))
    draw_text(frame, f"Stage: {stage}", (right_x, 34), color, 0.72, 2)
    draw_text(
        frame,
        f"Closed-loop Triggered: {triggered}   Progress: {progress * 100:05.1f}%",
        (right_x, 68),
        (230, 235, 240),
        0.56,
        1,
    )

    draw_text(
        frame,
        "Current trigger mode: PoC rule-based video progress trigger",
        (24, header_h - 16),
        (180, 195, 210),
        0.46,
        1,
    )


def draw_zones(frame, stage: str) -> Dict[str, int]:
    """Draw safe zone, warning zone label, and danger boundary."""
    h, w = frame.shape[:2]
    header_h = max(108, int(h * 0.14))

    safe_left = int(w * 0.12)
    safe_top = max(header_h + 28, int(h * 0.22))
    safe_right = int(w * 0.78)
    safe_bottom = int(h * 0.88)
    danger_x = int(w * 0.80)
    warning_left = int(w * 0.66)

    green = STAGE_COLORS["SAFE"]
    yellow = STAGE_COLORS["WARNING"]
    red = STAGE_COLORS["DANGEROUS_ACTION"]

    cv2.rectangle(frame, (safe_left, safe_top), (safe_right, safe_bottom), green, 2)
    draw_text(frame, "Green Box: Safe Zone", (safe_left + 8, safe_top + 24), green, 0.52, 1)

    warning_overlay = frame.copy()
    cv2.rectangle(
        warning_overlay,
        (warning_left, safe_top),
        (safe_right, safe_bottom),
        yellow,
        -1,
    )
    cv2.addWeighted(warning_overlay, 0.13, frame, 0.87, 0, frame)
    draw_text(
        frame,
        "Warning Zone",
        (warning_left + 8, safe_top + 52),
        yellow,
        0.52,
        1,
    )

    cv2.line(frame, (danger_x, safe_top), (danger_x, safe_bottom), red, 4)
    draw_text(
        frame,
        "Red Line: Danger Boundary",
        (max(16, danger_x - 260), min(h - 58, safe_bottom + 30)),
        red,
        0.52,
        1,
    )

    if stage == "DANGEROUS_ACTION":
        banner_top = max(safe_top + 72, h - 46)
        cv2.rectangle(frame, (0, banner_top), (w, h), red, -1)
        draw_text(
            frame,
            "Dangerous action stage reached: backend AI closed-loop runs once.",
            (24, h - 16),
            (255, 255, 255),
            0.56,
            1,
        )

    return {
        "safe_left": safe_left,
        "safe_top": safe_top,
        "safe_right": safe_right,
        "safe_bottom": safe_bottom,
        "danger_x": danger_x,
    }


def draw_tracking_point(frame, stage: str, zones: Dict[str, int]) -> None:
    """Draw the PoC simulated tracking point."""
    h, w = frame.shape[:2]
    color = STAGE_COLORS[stage]

    if stage == "SAFE":
        point_x = int(w * 0.46)
    elif stage == "WARNING":
        point_x = int(w * 0.69)
    else:
        point_x = int(w * 0.83)

    point_y = int((zones["safe_top"] + zones["safe_bottom"]) * 0.54)

    cv2.circle(frame, (point_x, point_y), 10, color, -1)
    cv2.circle(frame, (point_x, point_y), 20, color, 2)
    cv2.line(frame, (point_x - 28, point_y), (point_x + 28, point_y), color, 1)
    cv2.line(frame, (point_x, point_y - 28), (point_x, point_y + 28), color, 1)

    label_x = min(max(18, point_x - 150), w - 330)
    label_y = max(zones["safe_top"] + 86, point_y - 34)
    draw_text(
        frame,
        "PoC simulated tracking point",
        (label_x, label_y),
        color,
        0.52,
        1,
    )


def draw_status_overlay(frame, stage: str, progress: float, triggered: bool):
    """Draw all OpenCV overlays for the video demo."""
    draw_header(frame, stage, progress, triggered)
    zones = draw_zones(frame, stage)
    draw_tracking_point(frame, stage, zones)
    return frame


def print_demo_intro(video_path: Path, total_frames: int, fps: float) -> None:
    """Print a screen-recording friendly demo intro."""
    print("=" * 78)
    print("Demo Name: Dangerous Action Complete AI Closed-loop Demo")
    print(f"Video Path: {video_path}")
    print("Trigger Mode: PoC rule-based video progress trigger")
    print("Closed-loop Modules: State Recognition -> Risk Decision -> Voice Warning -> Parent Alert -> Event Log")
    print(f"Video Metadata: total_frames={total_frames}, fps={fps:.2f}")
    print("Stage Rules: 0%-35%=SAFE, 35%-70%=WARNING, 70%-100%=DANGEROUS_ACTION")
    print("Note: Current target point is simulated; this is not real visual model detection.")
    print("Upgrade path: pose detection / object detection / multimodal fusion.")
    print("Press Q in the video window to exit.")
    print("=" * 78)


def print_stage_change(progress: float, stage: str) -> None:
    """Print stage transition logs."""
    print(f"[VIDEO_STAGE] progress={progress * 100:05.1f}% stage={stage}")


def trigger_danger_closed_loop(video_path: Path, progress: float) -> Dict[str, Any]:
    """Trigger the LangGraph dangerous-action closed loop once."""
    print("\n" + "=" * 78)
    print("[TRIGGER] DANGEROUS_ACTION stage reached")
    print("[TRIGGER] Current trigger mode: PoC rule-based video progress trigger")
    print("[TRIGGER] Backend AI closed-loop is executing now.")
    print("=" * 78)

    event = create_mock_danger_event(
        {
            "source": "video_demo",
            "description": (
                "Rule-based video progress entered the dangerous_action stage. "
                "This PoC trigger validates the AI safety closed-loop path."
            ),
            "trigger_mode": "rule_based_video_progress",
            "video_path": str(video_path),
            "video_progress_when_triggered": round(progress, 4),
        }
    )

    app = build_danger_action_graph()
    final_state = app.invoke(event)
    summary = build_summary(final_state)

    print_final_summary(summary)
    return summary


def print_final_summary(summary: Dict[str, Any]) -> None:
    """Print a structured closed-loop result summary."""
    print("\n>>> Closed-loop Structured Summary")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("=" * 78 + "\n")


def main() -> None:
    if not VIDEO_PATH.exists():
        print(f"[ERROR] Video file not found: {VIDEO_PATH}")
        print("Expected path: data/dangerous_test_video.mp4")
        return

    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        print(f"[ERROR] Failed to open video file: {VIDEO_PATH}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 25.0

    delay = max(1, int(1000 / fps))
    print_demo_intro(VIDEO_PATH, total_frames, fps)

    triggered = False
    last_stage = ""
    final_summary: Dict[str, Any] | None = None

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        progress = min(1.0, current_frame / total_frames) if total_frames > 0 else 0.0
        stage = get_stage(progress)

        if stage != last_stage:
            print_stage_change(progress, stage)
            last_stage = stage

        if stage == "DANGEROUS_ACTION" and not triggered:
            triggered = True
            final_summary = trigger_danger_closed_loop(VIDEO_PATH, progress)

        frame = draw_status_overlay(frame, stage, progress, triggered)

        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(delay) & 0xFF
        if key in (ord("q"), ord("Q")):
            break

    cap.release()
    cv2.destroyAllWindows()

    print("\nDemo playback finished.")
    if triggered:
        if final_summary:
            print(f"Event log file: {EVENT_LOG_FILE}")
        print("[SUCCESS] Video danger stage -> dangerous_action -> AI closed-loop completed.")
    else:
        print("[WARN] Video ended before DANGEROUS_ACTION was triggered.")


if __name__ == "__main__":
    main()
