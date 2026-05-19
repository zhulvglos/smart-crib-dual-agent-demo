"""
demo_danger_action_detector.py

YOLO-assisted Dangerous Action Demo.

This demo uses a pretrained YOLO person detector to automatically locate a
person-like target in each video frame, then applies a rule-based crib boundary
judgment to trigger the existing dangerous_action AI closed loop.

It is not a production-grade baby danger detector. It is a stronger PoC than
manual ROI tracking because the target box comes from a detector instead of a
fixed initialization box.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2

from demo_danger_action import (
    EVENT_LOG_FILE,
    build_danger_action_graph,
    build_summary,
    create_mock_danger_event,
)


CONFIG_PATH = Path("config/demo_detector_config.json")
DEFAULT_VIDEO_PATH = Path("data/dangerous_test1.mp4")
WINDOW_NAME = "AI Crib Safety Demo - YOLO Person Detector"

STAGE_COLORS = {
    "SAFE": (70, 210, 90),
    "WARNING": (0, 190, 255),
    "DANGEROUS_ACTION": (50, 70, 255),
}


Rect = Tuple[int, int, int, int]
Point = Tuple[int, int]


@dataclass
class Detection:
    bbox: Rect
    class_name: str
    confidence: float

    @property
    def center(self) -> Point:
        x, y, w, h = self.bbox
        return int(x + w / 2), int(y + h / 2)

    @property
    def right(self) -> int:
        x, _, w, _ = self.bbox
        return x + w

    @property
    def area(self) -> int:
        _, _, w, h = self.bbox
        return w * h


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ratio_rect_to_pixels(ratio_rect: Dict[str, float], width: int, height: int) -> Rect:
    x1 = int(ratio_rect["x1_ratio"] * width)
    y1 = int(ratio_rect["y1_ratio"] * height)
    x2 = int(ratio_rect["x2_ratio"] * width)
    y2 = int(ratio_rect["y2_ratio"] * height)
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(x1 + 1, min(width, x2))
    y2 = max(y1 + 1, min(height, y2))
    return x1, y1, x2 - x1, y2 - y1


def point_in_rect(point: Point, rect: Rect) -> bool:
    x, y = point
    rx, ry, rw, rh = rect
    return rx <= x <= rx + rw and ry <= y <= ry + rh


def bbox_iou(a: Rect, b: Rect) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def resolve_geometry(config: Dict[str, Any], width: int, height: int) -> Dict[str, Any]:
    return {
        "safe_zone": ratio_rect_to_pixels(config["safe_zone"], width, height),
        "warning_zone": ratio_rect_to_pixels(config["warning_zone"], width, height),
        "danger_boundary_x": int(config["danger_boundary"]["x_ratio"] * width),
    }


def load_yolo_model(model_path: str):
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency: ultralytics. Install it in this environment with "
            "`python -m pip install ultralytics`."
        ) from exc
    return YOLO(model_path)


def run_detector(model, frame, config: Dict[str, Any]) -> List[Detection]:
    model_cfg = config["model"]
    target_names = {name.lower() for name in model_cfg.get("target_class_names", ["person"])}
    min_conf = float(model_cfg.get("confidence_threshold", 0.25))
    image_size = int(model_cfg.get("image_size", 640))
    device = model_cfg.get("device", "cpu")
    iou_threshold = float(model_cfg.get("iou_threshold", 0.45))

    results = model.predict(
        source=frame,
        imgsz=image_size,
        conf=min_conf,
        iou=iou_threshold,
        device=device,
        verbose=False,
    )

    detections: List[Detection] = []
    if not results:
        return detections

    names = results[0].names
    boxes = results[0].boxes
    if boxes is None:
        return detections

    for box in boxes:
        class_id = int(box.cls[0].item())
        class_name = str(names.get(class_id, class_id)).lower()
        confidence = float(box.conf[0].item())
        if class_name not in target_names or confidence < min_conf:
            continue
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        detections.append(Detection((x1, y1, max(1, x2 - x1), max(1, y2 - y1)), class_name, confidence))

    return detections


def select_target(
    detections: List[Detection],
    previous: Optional[Detection],
    frame_size: Tuple[int, int],
    config: Dict[str, Any],
) -> Optional[Detection]:
    if not detections:
        return None

    height, width = frame_size
    selection_cfg = config.get("target_selection", {})
    min_area_ratio = float(selection_cfg.get("min_box_area_ratio", 0.01))
    min_area = width * height * min_area_ratio
    candidates = [d for d in detections if d.area >= min_area]
    if not candidates:
        candidates = detections

    if previous is None:
        return max(candidates, key=lambda d: (d.area, d.confidence))

    px, py = previous.center
    diagonal = (width * width + height * height) ** 0.5

    def score(det: Detection) -> float:
        cx, cy = det.center
        distance = (((cx - px) ** 2 + (cy - py) ** 2) ** 0.5) / diagonal
        return bbox_iou(det.bbox, previous.bbox) * 2.5 + det.confidence + (det.area / (width * height)) - distance

    return max(candidates, key=score)


def get_trigger_x(det: Detection, trigger_point: str) -> int:
    if trigger_point == "bbox_right":
        return det.right
    if trigger_point == "bbox_left":
        return det.bbox[0]
    return det.center[0]


def classify_stage(det: Optional[Detection], geometry: Dict[str, Any], config: Dict[str, Any]) -> str:
    if det is None:
        return "SAFE"

    trigger_point = config.get("danger_boundary", {}).get("trigger_point", "bbox_right")
    trigger_x = get_trigger_x(det, trigger_point)
    center = det.center
    warning_x = geometry["warning_zone"][0]

    if trigger_x >= geometry["danger_boundary_x"]:
        return "DANGEROUS_ACTION"
    if trigger_x >= warning_x or point_in_rect(center, geometry["warning_zone"]):
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
    if raw_stage == "WARNING" and current_stage == "WARNING":
        return "WARNING", warning_count, danger_count
    if raw_stage == "DANGEROUS_ACTION" and current_stage == "WARNING":
        return "WARNING", warning_count, danger_count
    return "SAFE", warning_count, danger_count


def draw_text(frame, text: str, origin: Point, color=(245, 245, 245), scale=0.56, thickness=1) -> None:
    x, y = origin
    cv2.putText(frame, text, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_header(frame, stage: str, triggered: bool, det: Optional[Detection], raw_stage: str, model_name: str) -> None:
    height, width = frame.shape[:2]
    color = STAGE_COLORS[stage]
    header_h = max(120, int(height * 0.16))
    target_text = "none"
    if det:
        cx, cy = det.center
        target_text = f"{det.class_name} {det.confidence:.2f} center=({cx}, {cy})"

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, header_h), (18, 22, 28), -1)
    cv2.addWeighted(overlay, 0.86, frame, 0.14, 0, frame)
    cv2.line(frame, (0, header_h), (width, header_h), color, 3)

    draw_text(frame, "AI Crib Safety Demo", (24, 34), (255, 255, 255), 0.82, 2)
    draw_text(frame, "Mode: YOLO Person Detection Assisted Demo", (24, 68), (210, 220, 230), 0.55, 1)
    draw_text(frame, "Pretrained detector + rule-based danger boundary", (24, header_h - 16), (180, 195, 210), 0.46, 1)

    right_x = max(24, int(width * 0.48))
    draw_text(frame, f"Stage: {stage}", (right_x, 34), color, 0.72, 2)
    draw_text(frame, f"Closed-loop Triggered: {triggered}   Raw: {raw_stage}", (right_x, 68), (230, 235, 240), 0.52, 1)
    draw_text(frame, f"Target: {target_text}", (right_x, 98), (230, 235, 240), 0.50, 1)
    draw_text(frame, f"Model: {model_name}", (right_x, header_h - 16), (190, 205, 220), 0.44, 1)


def draw_zones(frame, geometry: Dict[str, Any]) -> None:
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


def draw_detection(frame, det: Optional[Detection], stage: str, trigger_point: str) -> None:
    if det is None:
        draw_text(frame, "No person target detected", (24, frame.shape[0] - 24), (0, 180, 255), 0.62, 2)
        return

    color = STAGE_COLORS[stage]
    x, y, w, h = det.bbox
    cx, cy = det.center
    trigger_x = get_trigger_x(det, trigger_point)
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
    cv2.circle(frame, (cx, cy), 7, color, -1)
    cv2.circle(frame, (cx, cy), 16, color, 2)
    cv2.line(frame, (trigger_x, y), (trigger_x, y + h), color, 2)
    draw_text(frame, f"YOLO Target Box: {det.class_name} {det.confidence:.2f}", (x, max(22, y - 10)), color, 0.50, 1)


def draw_banner(frame, stage: str, triggered: bool) -> None:
    if stage != "DANGEROUS_ACTION":
        return
    height, width = frame.shape[:2]
    red = STAGE_COLORS["DANGEROUS_ACTION"]
    cv2.rectangle(frame, (0, height - 58), (width, height), red, -1)
    draw_text(frame, "Dangerous Action Detected", (24, height - 34), (255, 255, 255), 0.68, 2)
    draw_text(frame, "AI Closed-loop Activated" if triggered else "AI Closed-loop Pending", (24, height - 12), (255, 255, 255), 0.52, 1)


def draw_overlay(
    frame,
    stage: str,
    raw_stage: str,
    triggered: bool,
    det: Optional[Detection],
    geometry: Dict[str, Any],
    config: Dict[str, Any],
    warning_count: int,
    danger_count: int,
) -> None:
    draw_header(frame, stage, triggered, det, raw_stage, config["model"].get("path", "yolo11n.pt"))
    draw_zones(frame, geometry)
    draw_detection(frame, det, stage, config.get("danger_boundary", {}).get("trigger_point", "bbox_right"))
    draw_banner(frame, stage, triggered)
    if config.get("display", {}).get("show_debug_text", True):
        draw_text(frame, f"warning_frames={warning_count}  danger_frames={danger_count}", (24, frame.shape[0] - 76), (220, 225, 230), 0.48, 1)


def trigger_danger_closed_loop(
    video_path: Path,
    frame_index: int,
    det: Detection,
    danger_boundary_x: int,
    model_name: str,
) -> Dict[str, Any]:
    print("\n" + "=" * 84)
    print("[TRIGGER] DANGEROUS_ACTION confirmed by YOLO person detection + boundary rule")
    print("[TRIGGER] This is a pretrained-detector assisted demo, not product-grade baby pose recognition.")
    print("=" * 84)

    x, y, w, h = det.bbox
    event = create_mock_danger_event(
        {
            "source": "yolo_person_detector_video_demo",
            "description": (
                "A pretrained YOLO person detector found the target crossing the configured "
                "danger boundary for consecutive frames."
            ),
            "trigger_mode": "yolo_person_detection_boundary",
            "video_path": str(video_path),
            "frame_index": frame_index,
            "target_center": {"x": det.center[0], "y": det.center[1]},
            "target_bbox": {"x": x, "y": y, "w": w, "h": h},
            "danger_boundary_x": int(danger_boundary_x),
            "detector_model": model_name,
            "detection_class": det.class_name,
            "detection_confidence": round(det.confidence, 4),
            "confidence": round(max(0.8, det.confidence), 4),
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
            "target_bbox": final_state.get("target_bbox"),
            "danger_boundary_x": final_state.get("danger_boundary_x"),
            "detector_model": final_state.get("detector_model"),
            "detection_confidence": final_state.get("detection_confidence"),
        }
    )
    print("\n>>> YOLO Detector Closed-loop Summary")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("=" * 84 + "\n")
    return summary


def print_intro(config: Dict[str, Any], video_path: Path, total_frames: int, fps: float) -> None:
    print("=" * 84)
    print("Demo Name: YOLO-assisted Dangerous Action Demo")
    print("Mode: Pretrained YOLO person detector + rule-based danger boundary")
    print(f"Video Path: {video_path}")
    print(f"Model: {config['model'].get('path', 'yolo11n.pt')}")
    print("Important Notice:")
    print("  This is not product-grade baby danger detection.")
    print("  It uses a pretrained person detector. Accuracy depends on camera angle, occlusion, lighting, and scale.")
    print("  Danger is judged by configured boundary rules, not by a trained baby-risk classifier.")
    print(f"Video Metadata: total_frames={total_frames}, fps={fps:.2f}")
    print("=" * 84)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO-assisted dangerous-action boundary demo.")
    parser.add_argument("--config", default=str(CONFIG_PATH), help="Path to detector config JSON.")
    parser.add_argument("--video", default=None, help="Override video_path from config.")
    parser.add_argument("--model", default=None, help="Override YOLO model path, e.g. yolo11n.pt or yolo11n-pose.pt.")
    parser.add_argument("--speed", type=float, default=None, help="Playback speed multiplier.")
    parser.add_argument("--imgsz", type=int, default=None, help="YOLO inference image size.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(Path(args.config))
    if args.model:
        config["model"]["path"] = args.model
    if args.imgsz:
        config["model"]["image_size"] = args.imgsz

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
        print("[ERROR] Failed to read first frame.")
        cap.release()
        return

    height, width = first_frame.shape[:2]
    geometry = resolve_geometry(config, width, height)
    model_name = config["model"].get("path", "yolo11n.pt")
    model = load_yolo_model(model_name)

    playback_speed = args.speed or float(config.get("display", {}).get("playback_speed", 1.0))
    playback_speed = max(0.1, playback_speed)
    delay = max(1, int((1000 / fps) / playback_speed))
    detect_every_n = max(1, int(config["model"].get("detect_every_n_frames", 1)))
    warning_confirm_frames = int(config.get("debounce", {}).get("warning_confirm_frames", 3))
    danger_confirm_frames = int(config.get("debounce", {}).get("danger_confirm_frames", 5))
    lost_keep_frames = int(config.get("target_selection", {}).get("lost_keep_frames", 8))

    print_intro(config, video_path, total_frames, fps)
    print(f"[GEOMETRY] danger_boundary_x={geometry['danger_boundary_x']}")
    print(f"[PLAYBACK] speed={playback_speed:.2f}x delay_ms={delay}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    previous_target: Optional[Detection] = None
    current_target: Optional[Detection] = None
    lost_frames = 0
    warning_count = 0
    danger_count = 0
    displayed_stage = "SAFE"
    last_logged_stage = ""
    raw_stage = "SAFE"
    triggered = False
    final_summary: Optional[Dict[str, Any]] = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_index = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if frame_index % detect_every_n == 0 or current_target is None:
            detections = run_detector(model, frame, config)
            selected = select_target(detections, previous_target, frame.shape[:2], config)
            if selected is not None:
                current_target = selected
                previous_target = selected
                lost_frames = 0
            else:
                lost_frames += 1
                if lost_frames > lost_keep_frames:
                    current_target = None

        raw_stage = classify_stage(current_target, geometry, config)
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
            center_text = f"{current_target.center}" if current_target else "n/a"
            conf_text = f"{current_target.confidence:.2f}" if current_target else "n/a"
            print(
                f"[DETECTOR_STAGE] frame={frame_index} stage={displayed_stage} raw={raw_stage} "
                f"target_center={center_text} conf={conf_text} "
                f"warning_count={warning_count} danger_count={danger_count}"
            )
            last_logged_stage = displayed_stage

        if frame_index % 30 == 0:
            center_text = f"{current_target.center}" if current_target else "n/a"
            print(
                f"[DETECTOR_DEBUG] frame={frame_index} raw={raw_stage} target_center={center_text} "
                f"lost_frames={lost_frames} warning_count={warning_count} danger_count={danger_count}"
            )

        if displayed_stage == "DANGEROUS_ACTION" and not triggered and current_target is not None:
            triggered = True
            final_summary = trigger_danger_closed_loop(
                video_path,
                frame_index,
                current_target,
                geometry["danger_boundary_x"],
                model_name,
            )

        if config.get("display", {}).get("show_overlay", True):
            draw_overlay(
                frame,
                displayed_stage,
                raw_stage,
                triggered,
                current_target,
                geometry,
                config,
                warning_count,
                danger_count,
            )

        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(delay) & 0xFF
        if key in (ord("q"), ord("Q")):
            break

    cap.release()
    cv2.destroyAllWindows()

    print("\nYOLO-assisted detector demo playback finished.")
    if triggered:
        if final_summary:
            print(f"Event log file: {EVENT_LOG_FILE}")
        print("[SUCCESS] YOLO person detector boundary -> dangerous_action -> AI closed-loop completed.")
    else:
        print("[WARN] Video ended before detector boundary trigger was confirmed.")


if __name__ == "__main__":
    main()
