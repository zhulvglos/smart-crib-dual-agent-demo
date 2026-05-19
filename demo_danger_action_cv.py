"""
demo_danger_action_cv.py

CV-assisted Dangerous Action Demo.

Current version is CV-assisted PoC.
It uses OpenCV target tracking and rule-based danger boundary judgment.
It validates the AI closed-loop after a dangerous action event is detected.

This is not production-grade baby pose recognition.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2

from demo_danger_action import (
    EVENT_LOG_FILE,
    build_danger_action_graph,
    build_summary,
    create_mock_danger_event,
)


CONFIG_PATH = Path("config/demo_cv_config.json")
DEFAULT_VIDEO_PATH = Path("data/dangerous_test_video.mp4")
WINDOW_NAME = "AI Crib Safety Demo - CV-assisted PoC"

STAGE_COLORS = {
    "SAFE": (70, 210, 90),
    "WARNING": (0, 190, 255),
    "DANGEROUS_ACTION": (50, 70, 255),
}


Rect = Tuple[int, int, int, int]
Point = Tuple[int, int]


class TemplateMatchingTracker:
    """Small local-search template tracker for PoC demos."""

    def __init__(
        self,
        search_radius_ratio: float = 1.8,
        update_rate: float = 0.08,
        min_score: float = 0.35,
    ) -> None:
        self.search_radius_ratio = search_radius_ratio
        self.update_rate = update_rate
        self.min_score = min_score
        self.template = None
        self.bbox: Optional[Rect] = None

    def init(self, frame, bbox: Rect) -> bool:
        x, y, w, h = bbox
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.template = gray[y : y + h, x : x + w].copy()
        self.bbox = bbox
        return self.template.size > 0

    def update(self, frame) -> tuple[bool, Rect]:
        if self.template is None or self.bbox is None:
            return False, (0, 0, 0, 0)

        height, width = frame.shape[:2]
        x, y, w, h = self.bbox
        radius = int(max(w, h) * self.search_radius_ratio)
        sx1 = max(0, x - radius)
        sy1 = max(0, y - radius)
        sx2 = min(width, x + w + radius)
        sy2 = min(height, y + h + radius)

        search = cv2.cvtColor(frame[sy1:sy2, sx1:sx2], cv2.COLOR_BGR2GRAY)
        if search.shape[0] < h or search.shape[1] < w:
            return False, self.bbox

        result = cv2.matchTemplate(search, self.template, cv2.TM_CCOEFF_NORMED)
        _, max_score, _, max_loc = cv2.minMaxLoc(result)
        if max_score < self.min_score:
            return False, self.bbox

        nx = sx1 + max_loc[0]
        ny = sy1 + max_loc[1]
        matched = cv2.cvtColor(frame[ny : ny + h, nx : nx + w], cv2.COLOR_BGR2GRAY)
        if matched.shape == self.template.shape and self.update_rate > 0:
            self.template = cv2.addWeighted(
                self.template,
                1.0 - self.update_rate,
                matched,
                self.update_rate,
                0,
            )

        self.bbox = (nx, ny, w, h)
        return True, self.bbox


class OpenCVTrackerAdapter:
    """Normalize OpenCV tracker APIs to the local tracker interface."""

    def __init__(self, tracker) -> None:
        self.tracker = tracker

    def init(self, frame, bbox: Rect) -> bool:
        result = self.tracker.init(frame, bbox)
        return True if result is None else bool(result)

    def update(self, frame) -> tuple[bool, Rect]:
        ok, bbox = self.tracker.update(frame)
        return bool(ok), tuple(int(v) for v in bbox)


def load_config(config_path: Path = CONFIG_PATH) -> Dict[str, Any]:
    """Load the CV demo config."""
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config file: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ratio_rect_to_pixels(ratio_rect: Dict[str, float], width: int, height: int) -> Rect:
    """Convert x1/y1/x2/y2 ratio coordinates into an OpenCV rectangle."""
    x1 = int(ratio_rect["x1_ratio"] * width)
    y1 = int(ratio_rect["y1_ratio"] * height)
    x2 = int(ratio_rect["x2_ratio"] * width)
    y2 = int(ratio_rect["y2_ratio"] * height)
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(x1 + 1, min(width, x2))
    y2 = max(y1 + 1, min(height, y2))
    return x1, y1, x2 - x1, y2 - y1


def rect_center(rect: Rect) -> Point:
    """Return the center point of an OpenCV rectangle."""
    x, y, w, h = rect
    return int(x + w / 2), int(y + h / 2)


def point_in_rect(point: Point, rect: Rect) -> bool:
    """Return whether a point is inside an OpenCV rectangle."""
    x, y = point
    rx, ry, rw, rh = rect
    return rx <= x <= rx + rw and ry <= y <= ry + rh


def create_tracker(method: str, tracking_config: Dict[str, Any]):
    """Create a lightweight OpenCV tracker without adding new dependencies."""
    normalized = method.strip().upper()
    if normalized in {"TEMPLATE_MATCHING", "TEMPLATE", "MATCH_TEMPLATE"}:
        return TemplateMatchingTracker(
            search_radius_ratio=float(tracking_config.get("template_search_radius_ratio", 1.8)),
            update_rate=float(tracking_config.get("template_update_rate", 0.08)),
            min_score=float(tracking_config.get("template_min_score", 0.35)),
        )

    factories = {
        "MIL": "TrackerMIL_create",
        "DASIAMRPN": "TrackerDaSiamRPN_create",
        "NANO": "TrackerNano_create",
    }

    factory_name = factories.get(normalized, "TrackerMIL_create")
    factory = getattr(cv2, factory_name, None)
    if factory is None:
        print(f"[WARN] OpenCV tracker '{method}' is unavailable. Falling back to MIL.")
        factory = cv2.TrackerMIL_create
    return OpenCVTrackerAdapter(factory())


def initialize_roi(frame, config: Dict[str, Any], force_manual: bool = False) -> Optional[Rect]:
    """Initialize target ROI from config, or use manual selection if omitted."""
    height, width = frame.shape[:2]
    roi_config = config.get("tracking", {}).get("initial_roi_ratio")

    if roi_config and not force_manual:
        return ratio_rect_to_pixels(roi_config, width, height)

    print("[ROI] Please select the target ROI for this video, then press ENTER or SPACE.")
    print("[ROI] Tip: select the baby's head/upper body, not the whole crib.")
    selected = cv2.selectROI(WINDOW_NAME, frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow(WINDOW_NAME)
    x, y, w, h = [int(v) for v in selected]
    if w <= 0 or h <= 0:
        return None
    return x, y, w, h


def resolve_geometry(config: Dict[str, Any], width: int, height: int) -> Dict[str, Any]:
    """Resolve ratio-based config geometry to pixel coordinates."""
    safe_zone = ratio_rect_to_pixels(config["safe_zone"], width, height)
    warning_zone = ratio_rect_to_pixels(config["warning_zone"], width, height)
    danger_boundary_x = int(config["danger_boundary"]["x_ratio"] * width)
    return {
        "safe_zone": safe_zone,
        "warning_zone": warning_zone,
        "danger_boundary_x": danger_boundary_x,
    }


def classify_raw_stage(
    center: Optional[Point],
    geometry: Dict[str, Any],
    tracking_active: bool,
) -> str:
    """Classify one frame before debounce."""
    if not tracking_active or center is None:
        return "SAFE"

    if center[0] >= geometry["danger_boundary_x"]:
        return "DANGEROUS_ACTION"

    if point_in_rect(center, geometry["warning_zone"]):
        return "WARNING"

    if point_in_rect(center, geometry["safe_zone"]):
        return "SAFE"

    return "WARNING"


def apply_debounce(
    raw_stage: str,
    current_stage: str,
    warning_count: int,
    danger_count: int,
    warning_confirm_frames: int,
    danger_confirm_frames: int,
) -> tuple[str, int, int]:
    """Apply warning and danger frame-count confirmation."""
    if raw_stage == "DANGEROUS_ACTION":
        danger_count += 1
        warning_count = 0
    elif raw_stage == "WARNING":
        warning_count += 1
        danger_count = 0
    else:
        warning_count = 0
        danger_count = 0

    if danger_count >= danger_confirm_frames:
        return "DANGEROUS_ACTION", warning_count, danger_count
    if warning_count >= warning_confirm_frames:
        return "WARNING", warning_count, danger_count
    if raw_stage == "DANGEROUS_ACTION" and current_stage == "WARNING":
        return "WARNING", warning_count, danger_count
    return "SAFE", warning_count, danger_count


def draw_text(
    frame,
    text: str,
    origin: Point,
    color: tuple[int, int, int] = (245, 245, 245),
    scale: float = 0.56,
    thickness: int = 1,
) -> None:
    """Draw readable antialiased text with a subtle shadow."""
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


def draw_header(
    frame,
    stage: str,
    triggered: bool,
    tracking_active: bool,
    center: Optional[Point],
) -> None:
    """Draw the top status panel."""
    height, width = frame.shape[:2]
    color = STAGE_COLORS[stage]
    header_h = max(116, int(height * 0.15))
    tracking_text = "active" if tracking_active else "lost"
    center_text = f"{center[0]}, {center[1]}" if center else "n/a"

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, header_h), (18, 22, 28), -1)
    cv2.addWeighted(overlay, 0.86, frame, 0.14, 0, frame)
    cv2.line(frame, (0, header_h), (width, header_h), color, 3)

    draw_text(frame, "AI Crib Safety Demo", (24, 34), (255, 255, 255), 0.82, 2)
    draw_text(frame, "Mode: CV-assisted PoC", (24, 68), (210, 220, 230), 0.58, 1)
    draw_text(
        frame,
        "OpenCV target tracking + rule-based danger boundary trigger",
        (24, header_h - 16),
        (180, 195, 210),
        0.46,
        1,
    )

    right_x = max(24, int(width * 0.47))
    draw_text(frame, f"Stage: {stage}", (right_x, 34), color, 0.72, 2)
    draw_text(
        frame,
        f"Closed-loop Triggered: {triggered}   Tracking: {tracking_text}",
        (right_x, 68),
        (230, 235, 240),
        0.54,
        1,
    )
    draw_text(frame, f"Target center: {center_text}", (right_x, 98), (230, 235, 240), 0.54, 1)


def draw_zones(frame, geometry: Dict[str, Any]) -> None:
    """Draw configured safe/warning zones and danger boundary."""
    safe_zone = geometry["safe_zone"]
    warning_zone = geometry["warning_zone"]
    danger_x = geometry["danger_boundary_x"]
    height, width = frame.shape[:2]
    green = STAGE_COLORS["SAFE"]
    yellow = STAGE_COLORS["WARNING"]
    red = STAGE_COLORS["DANGEROUS_ACTION"]

    sx, sy, sw, sh = safe_zone
    wx, wy, ww, wh = warning_zone

    cv2.rectangle(frame, (sx, sy), (sx + sw, sy + sh), green, 2)
    draw_text(frame, "Green Box: Safe Zone", (sx + 8, sy + 24), green, 0.50, 1)

    warning_overlay = frame.copy()
    cv2.rectangle(warning_overlay, (wx, wy), (wx + ww, wy + wh), yellow, -1)
    cv2.addWeighted(warning_overlay, 0.14, frame, 0.86, 0, frame)
    cv2.rectangle(frame, (wx, wy), (wx + ww, wy + wh), yellow, 2)
    draw_text(frame, "Yellow Area: Warning Zone", (wx + 8, wy + 52), yellow, 0.50, 1)

    cv2.line(frame, (danger_x, sy), (danger_x, min(height - 1, sy + sh)), red, 4)
    label_x = max(16, min(width - 290, danger_x - 190))
    draw_text(frame, "Red Line: Danger Boundary", (label_x, min(height - 54, sy + sh + 30)), red, 0.50, 1)


def draw_target(frame, bbox: Optional[Rect], center: Optional[Point], tracking_active: bool, stage: str) -> None:
    """Draw target bounding box and center point."""
    if not tracking_active or bbox is None or center is None:
        draw_text(frame, "Tracking lost", (24, frame.shape[0] - 24), (0, 180, 255), 0.62, 2)
        return

    color = STAGE_COLORS[stage]
    x, y, w, h = bbox
    cx, cy = center
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    cv2.circle(frame, center, 7, color, -1)
    cv2.circle(frame, center, 16, color, 2)
    cv2.line(frame, (cx - 22, cy), (cx + 22, cy), color, 1)
    cv2.line(frame, (cx, cy - 22), (cx, cy + 22), color, 1)
    draw_text(frame, "Target Bounding Box", (x, max(22, y - 10)), color, 0.48, 1)


def draw_danger_banner(frame, stage: str, triggered: bool) -> None:
    """Draw a bottom danger banner once danger is confirmed."""
    if stage != "DANGEROUS_ACTION":
        return

    height, width = frame.shape[:2]
    red = STAGE_COLORS["DANGEROUS_ACTION"]
    cv2.rectangle(frame, (0, height - 58), (width, height), red, -1)
    draw_text(frame, "Dangerous Action Detected", (24, height - 34), (255, 255, 255), 0.68, 2)
    status = "AI Closed-loop Activated" if triggered else "AI Closed-loop Pending"
    draw_text(frame, status, (24, height - 12), (255, 255, 255), 0.52, 1)


def draw_overlay(
    frame,
    stage: str,
    triggered: bool,
    tracking_active: bool,
    bbox: Optional[Rect],
    center: Optional[Point],
    geometry: Dict[str, Any],
    show_debug_text: bool,
    warning_count: int,
    danger_count: int,
) -> None:
    """Draw all display overlays."""
    draw_header(frame, stage, triggered, tracking_active, center)
    draw_zones(frame, geometry)
    draw_target(frame, bbox, center, tracking_active, stage)
    draw_danger_banner(frame, stage, triggered)

    if show_debug_text:
        debug = f"warning_frames={warning_count}  danger_frames={danger_count}"
        draw_text(frame, debug, (24, frame.shape[0] - 76), (220, 225, 230), 0.48, 1)


def print_intro(config: Dict[str, Any], video_path: Path, total_frames: int, fps: float) -> None:
    """Print the startup message."""
    print("=" * 82)
    print("Demo Name: CV-assisted Dangerous Action Demo")
    print("Mode: CV-assisted PoC")
    print(f"Video Path: {video_path}")
    print("Trigger Mode: target tracking + danger boundary")
    print("Important Notice:")
    print("  This is not production-grade baby pose recognition.")
    print("  It is an OpenCV-assisted PoC for validating the dangerous action closed-loop.")
    print("  For a new video, reselect the target ROI and recalibrate the boundary zones.")
    print(f"Tracking Method: {config.get('tracking', {}).get('method', 'MIL')}")
    print(f"Video Metadata: total_frames={total_frames}, fps={fps:.2f}")
    print("Press Q in the video window to exit.")
    print("=" * 82)


def print_stage_change(
    frame_index: int,
    stage: str,
    center: Optional[Point],
    warning_count: int,
    danger_count: int,
) -> None:
    """Print a stage transition line."""
    center_text = f"({center[0]}, {center[1]})" if center else "n/a"
    print(
        f"[CV_STAGE] frame={frame_index} stage={stage} target_center={center_text} "
        f"warning_count={warning_count} danger_count={danger_count}"
    )


def trigger_danger_closed_loop(
    video_path: Path,
    frame_index: int,
    center: Point,
    danger_boundary_x: int,
) -> Dict[str, Any]:
    """Trigger the dangerous-action LangGraph closed loop."""
    print("\n" + "=" * 82)
    print("[TRIGGER] DANGEROUS_ACTION confirmed by CV-assisted boundary rule")
    print("[TRIGGER] Current version: CV-assisted PoC")
    print("[TRIGGER] OpenCV target tracking + rule-based danger boundary judgment")
    print("=" * 82)

    event = create_mock_danger_event(
        {
            "source": "cv_assisted_video_demo",
            "description": (
                "OpenCV target tracking center crossed the configured danger boundary "
                "for consecutive frames in this CV-assisted PoC."
            ),
            "trigger_mode": "cv_target_tracking_boundary",
            "video_path": str(video_path),
            "frame_index": frame_index,
            "target_center": {"x": int(center[0]), "y": int(center[1])},
            "danger_boundary_x": int(danger_boundary_x),
            "confidence": 0.86,
        }
    )

    app = build_danger_action_graph()
    final_state = app.invoke(event)

    summary = build_summary(final_state)
    summary.update(
        {
            "source": final_state.get("source"),
            "trigger_mode": final_state.get("trigger_mode"),
            "frame_index": final_state.get("frame_index"),
            "target_center": final_state.get("target_center"),
            "danger_boundary_x": final_state.get("danger_boundary_x"),
        }
    )

    print("\n>>> CV-assisted Closed-loop Summary")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("=" * 82 + "\n")

    return summary


def parse_args() -> argparse.Namespace:
    """Parse command-line options for video swaps and ROI calibration."""
    parser = argparse.ArgumentParser(
        description="CV-assisted PoC: OpenCV target tracking + rule-based danger boundary trigger."
    )
    parser.add_argument("--config", default=str(CONFIG_PATH), help="Path to demo_cv_config.json.")
    parser.add_argument("--video", default=None, help="Override video_path from config.")
    parser.add_argument(
        "--select-roi",
        action="store_true",
        help="Manually select target ROI on the first frame. Recommended when changing videos.",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=None,
        help="Playback speed multiplier. Example: 2.0 for faster preview.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(Path(args.config))
    video_path = Path(args.video or config.get("video_path") or DEFAULT_VIDEO_PATH)

    if not video_path.exists():
        print(f"[ERROR] Video file not found: {video_path}")
        return

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] Failed to open video file: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    ret, first_frame = cap.read()
    if not ret:
        print("[ERROR] Failed to read the first video frame.")
        cap.release()
        return

    height, width = first_frame.shape[:2]
    geometry = resolve_geometry(config, width, height)
    initial_roi = initialize_roi(first_frame, config, force_manual=args.select_roi)
    if initial_roi is None:
        print("[ERROR] Target ROI initialization was cancelled.")
        cap.release()
        return

    tracking_config = config.get("tracking", {})
    tracker_method = tracking_config.get("method", "template_matching")
    tracker = create_tracker(tracker_method, tracking_config)
    tracker.init(first_frame, initial_roi)

    warning_confirm_frames = int(config.get("tracking", {}).get("warning_confirm_frames", 5))
    danger_confirm_frames = int(config.get("tracking", {}).get("danger_confirm_frames", 8))
    show_overlay = bool(config.get("display", {}).get("show_overlay", True))
    show_debug_text = bool(config.get("display", {}).get("show_debug_text", True))
    playback_speed = args.speed or float(config.get("display", {}).get("playback_speed", 1.0))
    playback_speed = max(0.1, playback_speed)
    delay = max(1, int((1000 / fps) / playback_speed))

    print_intro(config, video_path, total_frames, fps)
    print(f"[ROI] initial_bbox={initial_roi}")
    print(f"[GEOMETRY] danger_boundary_x={geometry['danger_boundary_x']}")
    print(f"[PLAYBACK] speed={playback_speed:.2f}x delay_ms={delay}")
    if not args.select_roi and config.get("tracking", {}).get("initial_roi_ratio"):
        print("[NOTICE] Using config initial_roi_ratio. If you changed videos, run with --select-roi.")

    warning_count = 0
    danger_count = 0
    displayed_stage = "SAFE"
    last_logged_stage = ""
    triggered = False
    final_summary: Optional[Dict[str, Any]] = None

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_index = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        tracking_active, bbox_raw = tracker.update(frame)
        bbox: Optional[Rect] = None
        center: Optional[Point] = None

        if tracking_active:
            bbox = tuple(int(v) for v in bbox_raw)
            center = rect_center(bbox)

        raw_stage = classify_raw_stage(center, geometry, tracking_active)
        displayed_stage, warning_count, danger_count = apply_debounce(
            raw_stage,
            displayed_stage,
            warning_count,
            danger_count,
            warning_confirm_frames,
            danger_confirm_frames,
        )
        if triggered:
            displayed_stage = "DANGEROUS_ACTION"

        if displayed_stage != last_logged_stage:
            print_stage_change(frame_index, displayed_stage, center, warning_count, danger_count)
            last_logged_stage = displayed_stage

        if frame_index % 30 == 0:
            center_text = f"({center[0]}, {center[1]})" if center else "n/a"
            print(
                f"[CV_DEBUG] frame={frame_index} raw_stage={raw_stage} "
                f"target_center={center_text} warning_count={warning_count} danger_count={danger_count}"
            )

        if displayed_stage == "DANGEROUS_ACTION" and not triggered and center is not None:
            triggered = True
            final_summary = trigger_danger_closed_loop(
                video_path,
                frame_index,
                center,
                geometry["danger_boundary_x"],
            )

        if show_overlay:
            draw_overlay(
                frame,
                displayed_stage,
                triggered,
                tracking_active,
                bbox,
                center,
                geometry,
                show_debug_text,
                warning_count,
                danger_count,
            )

        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(delay) & 0xFF
        if key in (ord("q"), ord("Q")):
            break

    cap.release()
    cv2.destroyAllWindows()

    print("\nCV-assisted demo playback finished.")
    if triggered:
        if final_summary:
            print(f"Event log file: {EVENT_LOG_FILE}")
        print("[SUCCESS] CV-assisted danger boundary -> dangerous_action -> AI closed-loop completed.")
    else:
        print("[WARN] Video ended before the CV-assisted danger boundary trigger was confirmed.")


if __name__ == "__main__":
    main()
