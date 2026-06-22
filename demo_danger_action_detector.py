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
import numpy as np

from demo_danger_action import (
    EVENT_LOG_FILE,
    build_danger_action_graph,
    build_summary,
    create_mock_danger_event,
)

from crib_detector import (
    CribDetector,
    CribGeometry,
    save_crib_config,
    load_crib_config,
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

COCO_KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]


@dataclass
class Detection:
    bbox: Rect
    class_name: str
    confidence: float
    keypoints: Optional[np.ndarray] = None  # YOLO-pose 关键点 [N, 3] (x, y, conf)

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

    def get_body_key_points(self) -> List[Point]:
        """提取身体关键部位坐标，用于更精确的安全判断。"""
        points: List[Point] = []

        # COCO keypoint indices for pose model
        # 0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear
        # 5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow
        # 9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip
        # 13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle
        if self.keypoints is not None and len(self.keypoints) > 0:
            kpts = self.keypoints
            # 头部中心（鼻子或双眼中心）
            if kpts[0][2] > 0.3:
                points.append((int(kpts[0][0]), int(kpts[0][1])))

            # 肩膀中心
            if kpts[5][2] > 0.3 and kpts[6][2] > 0.3:
                shoulder_x = int((kpts[5][0] + kpts[6][0]) / 2)
                shoulder_y = int((kpts[5][1] + kpts[6][1]) / 2)
                points.append((shoulder_x, shoulder_y))
            elif kpts[5][2] > 0.3:
                points.append((int(kpts[5][0]), int(kpts[5][1])))
            elif kpts[6][2] > 0.3:
                points.append((int(kpts[6][0]), int(kpts[6][1])))

            # 臀部中心
            if kpts[11][2] > 0.3 and kpts[12][2] > 0.3:
                hip_x = int((kpts[11][0] + kpts[12][0]) / 2)
                hip_y = int((kpts[11][1] + kpts[12][1]) / 2)
                points.append((hip_x, hip_y))

            # 脚踝（最低的位置，最能反映是否爬出床）
            ankle_candidates = []
            if kpts[15][2] > 0.3:
                ankle_candidates.append((int(kpts[15][0]), int(kpts[15][1])))
            if kpts[16][2] > 0.3:
                ankle_candidates.append((int(kpts[16][0]), int(kpts[16][1])))
            if ankle_candidates:
                # 取 y 值最大的（最靠下的）脚踝
                lowest_ankle = max(ankle_candidates, key=lambda p: p[1])
                points.append(lowest_ankle)

        return points

    def get_named_keypoints(self, min_confidence: float = 0.3) -> Dict[str, Tuple[int, int, float]]:
        """Return visible COCO keypoints keyed by semantic name."""
        if self.keypoints is None or len(self.keypoints) == 0:
            return {}

        result: Dict[str, Tuple[int, int, float]] = {}
        for index, name in enumerate(COCO_KEYPOINT_NAMES):
            if index >= len(self.keypoints):
                break
            x, y, confidence = self.keypoints[index]
            if confidence >= min_confidence:
                result[name] = (int(x), int(y), float(confidence))
        return result


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


def point_in_polygon(point: Point, polygon: np.ndarray) -> bool:
    """判断点是否在多边形内"""
    result = cv2.pointPolygonTest(polygon, (float(point[0]), float(point[1])), False)
    return result >= 0


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


def resolve_geometry_from_crib(
    crib_geometry: CribGeometry,
    width: int,
    height: int
) -> Dict[str, Any]:
    """从 CribGeometry 创建几何配置（向后兼容）"""
    safe_zone = cv2.boundingRect(crib_geometry.safe_contour)
    warning_zone = cv2.boundingRect(crib_geometry.warning_contour)
    
    return {
        "safe_zone": safe_zone,
        "warning_zone": warning_zone,
        "crib_geometry": crib_geometry,
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

    result = results[0]
    names = result.names
    boxes = result.boxes
    if boxes is None:
        return detections

    # 检查是否有姿态关键点（YOLO-pose）
    has_keypoints = hasattr(result, "keypoints") and result.keypoints is not None

    for i, box in enumerate(boxes):
        class_id = int(box.cls[0])
        class_name = str(names.get(class_id, class_id)).lower()
        confidence = float(box.conf[0])
        if class_name not in target_names or confidence < min_conf:
            continue
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]

        kpts = None
        if has_keypoints and i < len(result.keypoints):
            kpt_data = result.keypoints[i].data.cpu().numpy()
            if kpt_data.shape[0] > 0:
                kpts = kpt_data[0]  # [N, 3]

        detections.append(
            Detection(
                (x1, y1, max(1, x2 - x1), max(1, y2 - y1)),
                class_name,
                confidence,
                keypoints=kpts,
            )
        )

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


def is_near_crib_edge(
    point: Point,
    crib_contour: np.ndarray,
    threshold: float = 0.15,
) -> bool:
    """判断点是否接近床的边缘"""
    # 计算点到轮廓的距离
    distance = cv2.pointPolygonTest(crib_contour, (float(point[0]), float(point[1])), True)
    
    # 如果距离为负，说明点在轮廓外（危险）
    if distance < 0:
        return True
    
    # 如果距离很小，说明接近边缘
    x, y, w, h = cv2.boundingRect(crib_contour)
    max_dim = max(w, h)
    if distance < max_dim * threshold:
        return True
    
    return False


def _average_named_point(
    points: Dict[str, Tuple[int, int, float]],
    names: Tuple[str, str],
) -> Optional[Point]:
    available = [points[name] for name in names if name in points]
    if not available:
        return None
    return (
        int(sum(point[0] for point in available) / len(available)),
        int(sum(point[1] for point in available) / len(available)),
    )


def _signed_distance_to_contour(point: Point, contour: np.ndarray) -> float:
    return float(cv2.pointPolygonTest(contour, (float(point[0]), float(point[1])), True))


def analyze_pose_risk(
    det: Optional[Detection],
    geometry: Dict[str, Any],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Build an explainable pose-assisted crib-edge risk assessment."""
    if det is None:
        return {
            "mode": "no_target",
            "stage": "SAFE",
            "score": 0,
            "visible_keypoints": 0,
            "evidence": {},
        }

    pose_cfg = config.get("pose_risk", {})
    min_keypoint_conf = float(pose_cfg.get("min_keypoint_confidence", 0.3))
    min_visible = int(pose_cfg.get("min_visible_keypoints", 6))
    warning_score = int(pose_cfg.get("warning_score", 3))
    danger_score = int(pose_cfg.get("danger_score", 6))
    edge_ratio = float(pose_cfg.get("edge_distance_ratio", 0.08))
    lean_ratio = float(pose_cfg.get("lean_distance_ratio", 0.025))

    named = det.get_named_keypoints(min_keypoint_conf)
    visible_count = len(named)
    pose_available = visible_count >= min_visible

    trigger_point = config.get("danger_boundary", {}).get("trigger_point", "bbox_right")
    trigger_x = get_trigger_x(det, trigger_point)
    bbox_trigger = (trigger_x, det.center[1])

    evidence: Dict[str, Any] = {
        "bbox_near_edge": False,
        "bbox_outside": False,
        "one_wrist_near_rail": False,
        "both_wrists_near_rail": False,
        "shoulders_near_edge": False,
        "hips_near_edge": False,
        "head_near_edge": False,
        "upper_body_leaning_out": False,
        "pose_points_outside": 0,
    }

    if "crib_geometry" not in geometry:
        warning_x = geometry["warning_zone"][0]
        danger_x = geometry["danger_boundary_x"]
        evidence["bbox_near_edge"] = (
            trigger_x >= warning_x or point_in_rect(det.center, geometry["warning_zone"])
        )
        evidence["bbox_outside"] = trigger_x >= danger_x
        if evidence["bbox_outside"]:
            stage = "DANGEROUS_ACTION"
            score = danger_score
        elif evidence["bbox_near_edge"]:
            stage = "WARNING"
            score = warning_score
        else:
            stage = "SAFE"
            score = 0
        return {
            "mode": "bbox_fallback",
            "stage": stage,
            "score": min(10, score),
            "visible_keypoints": visible_count,
            "evidence": evidence,
        }

    crib_geometry = geometry["crib_geometry"]
    crib_contour = crib_geometry.crib_contour
    safe_contour = crib_geometry.safe_contour
    _, _, crib_w, crib_h = cv2.boundingRect(crib_contour)
    edge_distance = max(crib_w, crib_h) * edge_ratio
    lean_margin = max(crib_w, crib_h) * lean_ratio

    def is_pose_point_near_edge(point: Point) -> bool:
        return (
            not point_in_polygon(point, safe_contour)
            or _signed_distance_to_contour(point, crib_contour) <= edge_distance
        )

    trigger_crib_distance = _signed_distance_to_contour(bbox_trigger, crib_contour)
    evidence["bbox_near_edge"] = not point_in_polygon(bbox_trigger, safe_contour)
    evidence["bbox_outside"] = trigger_crib_distance < 0

    score = 0
    if evidence["bbox_near_edge"]:
        score += 1
    if evidence["bbox_outside"]:
        score += 2

    if pose_available:
        wrists = [
            (point[0], point[1])
            for name, point in named.items()
            if name in ("left_wrist", "right_wrist")
        ]
        wrist_near_count = sum(
            is_pose_point_near_edge(point)
            for point in wrists
        )
        evidence["one_wrist_near_rail"] = wrist_near_count >= 1
        evidence["both_wrists_near_rail"] = wrist_near_count >= 2
        if wrist_near_count == 1:
            score += 1
        elif wrist_near_count >= 2:
            score += 2

        shoulder_center = _average_named_point(
            named, ("left_shoulder", "right_shoulder")
        )
        hip_center = _average_named_point(named, ("left_hip", "right_hip"))
        nose = (named["nose"][0], named["nose"][1]) if "nose" in named else None

        shoulder_distance = None
        hip_distance = None
        if shoulder_center is not None:
            shoulder_distance = _signed_distance_to_contour(shoulder_center, crib_contour)
            evidence["shoulders_near_edge"] = is_pose_point_near_edge(shoulder_center)
            if evidence["shoulders_near_edge"]:
                score += 2

        if hip_center is not None:
            hip_distance = _signed_distance_to_contour(hip_center, crib_contour)
            evidence["hips_near_edge"] = is_pose_point_near_edge(hip_center)
            if evidence["hips_near_edge"]:
                score += 1

        if nose is not None:
            nose_distance = _signed_distance_to_contour(nose, crib_contour)
            evidence["head_near_edge"] = is_pose_point_near_edge(nose)
            if evidence["head_near_edge"]:
                score += 1

        if shoulder_distance is not None and hip_distance is not None:
            evidence["upper_body_leaning_out"] = (
                shoulder_distance + lean_margin < hip_distance
                and evidence["shoulders_near_edge"]
            )
            if evidence["upper_body_leaning_out"]:
                score += 2

        risk_names = (
            "nose",
            "left_shoulder",
            "right_shoulder",
            "left_wrist",
            "right_wrist",
            "left_hip",
            "right_hip",
        )
        outside_count = sum(
            _signed_distance_to_contour((named[name][0], named[name][1]), crib_contour) < 0
            for name in risk_names
            if name in named
        )
        evidence["pose_points_outside"] = outside_count
        if outside_count > 0:
            score += 2

    mode = "pose_assisted" if pose_available else "bbox_fallback"
    if mode == "bbox_fallback":
        if evidence["bbox_outside"]:
            stage = "DANGEROUS_ACTION"
            score = max(score, danger_score)
        elif evidence["bbox_near_edge"]:
            stage = "WARNING"
            score = max(score, warning_score)
        else:
            stage = "SAFE"
    elif score >= danger_score:
        stage = "DANGEROUS_ACTION"
    elif score >= warning_score:
        stage = "WARNING"
    else:
        stage = "SAFE"

    return {
        "mode": mode,
        "stage": stage,
        "score": min(10, score),
        "visible_keypoints": visible_count,
        "evidence": evidence,
    }


def classify_stage(
    det: Optional[Detection],
    geometry: Dict[str, Any],
    config: Dict[str, Any],
) -> str:
    """
    分类当前帧的安全状态。

    三区逻辑（使用 polygon-based crib 检测时）：
    - Safe Zone（绿框内）: 所有关键检查点都在 safe_contour 内部 → SAFE
    - Warning Zone（绿框与黄框之间）: 所有关键检查点在 crib_contour 内部，
      但至少有一个不在 safe_contour 内部 → WARNING
    - Danger Zone（黄框外）: 任一关键检查点在 crib_contour 外部 → DANGEROUS_ACTION
    """
    return str(analyze_pose_risk(det, geometry, config)["stage"])


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


def draw_header(frame, stage: str, triggered: bool, det: Optional[Detection], raw_stage: str, model_name: str, detection_method: str = "") -> None:
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
    if detection_method:
        draw_text(frame, f"Crib Detection: {detection_method}", (24, header_h - 16), (180, 195, 210), 0.46, 1)
    else:
        draw_text(frame, "Pretrained detector + rule-based danger boundary", (24, header_h - 16), (180, 195, 210), 0.46, 1)

    right_x = max(24, int(width * 0.48))
    draw_text(frame, f"Stage: {stage}", (right_x, 34), color, 0.72, 2)
    draw_text(frame, f"Closed-loop Triggered: {triggered}   Raw: {raw_stage}", (right_x, 68), (230, 235, 240), 0.52, 1)
    draw_text(frame, f"Target: {target_text}", (right_x, 98), (230, 235, 240), 0.50, 1)
    draw_text(frame, f"Model: {model_name}", (right_x, header_h - 16), (190, 205, 220), 0.44, 1)


def draw_zones(frame, geometry: Dict[str, Any]) -> None:
    height, width = frame.shape[:2]
    green = STAGE_COLORS["SAFE"]
    yellow = STAGE_COLORS["WARNING"]
    red = STAGE_COLORS["DANGEROUS_ACTION"]

    if "crib_geometry" in geometry:
        crib_geometry = geometry["crib_geometry"]
        cv2.polylines(frame, [crib_geometry.crib_contour], True, red, 3)
        cv2.polylines(frame, [crib_geometry.safe_contour], True, green, 2)
        cv2.polylines(frame, [crib_geometry.warning_contour], True, yellow, 2)
        
        draw_text(frame, "Green Box: Safe Zone", (20, height - 60), green, 0.5, 1)
        draw_text(frame, "Yellow/Red: Crib Boundary (Danger if outside)", (20, height - 40), yellow, 0.5, 1)
        draw_text(frame, "Between Green & Boundary: Warning Zone", (20, height - 20), yellow, 0.5, 1)
    else:
        safe_zone = geometry["safe_zone"]
        warning_zone = geometry["warning_zone"]
        danger_x = geometry["danger_boundary_x"]
        
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

    # 如果有关键点，绘制用于安全判断的身体关键部位
    if det.keypoints is not None and len(det.keypoints) > 0:
        # 绘制关键检查点
        pose_points = det.get_body_key_points()
        for i, pt in enumerate(pose_points):
            pt_color = (0, 255, 255) if i == 0 else (255, 200, 0)  # 头部黄色，其他橙色
            cv2.circle(frame, pt, 5, pt_color, -1)
            cv2.circle(frame, pt, 10, pt_color, 1)

    label = f"YOLO Target Box: {det.class_name} {det.confidence:.2f}"
    if det.keypoints is not None:
        label += " [pose]"
    draw_text(frame, label, (x, max(22, y - 10)), color, 0.50, 1)


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
    detection_method: str = "",
) -> None:
    draw_header(frame, stage, triggered, det, raw_stage, config["model"].get("path", "yolo11n.pt"), detection_method)
    draw_zones(frame, geometry)
    draw_detection(frame, det, stage, config.get("danger_boundary", {}).get("trigger_point", "bbox_right"))
    draw_banner(frame, stage, triggered)
    if config.get("display", {}).get("show_debug_text", True):
        draw_text(frame, f"warning_frames={warning_count}  danger_frames={danger_count}", (24, frame.shape[0] - 76), (220, 225, 230), 0.48, 1)


def trigger_danger_closed_loop(
    video_path: Path,
    frame_index: int,
    det: Detection,
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
    parser.add_argument("--auto-detect-crib", action="store_true", default=True, help="Auto-detect crib area (default: True)")
    parser.add_argument("--no-auto-detect", action="store_false", dest="auto_detect_crib", help="Disable auto-detect crib area")
    parser.add_argument("--safe-ratio", type=float, default=0.15, help="Safe zone shrink ratio")
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
    detection_method = ""
    
    if args.auto_detect_crib:
        print("[INFO] Attempting to auto-detect crib area...")
        try:
            saved_crib_geometry = load_crib_config(
                Path(args.config),
                str(video_path),
                width,
                height
            )
            if saved_crib_geometry is not None:
                print("[INFO] Using saved crib configuration")
                geometry = resolve_geometry_from_crib(saved_crib_geometry, width, height)
                detection_method = "saved"
            else:
                crib_detector = CribDetector()
                result = crib_detector.detect_crib(first_frame, safe_ratio=args.safe_ratio)
                
                if result.success and result.geometry is not None:
                    print(f"[INFO] Successfully detected crib using {result.method} method (confidence: {result.confidence:.2f})")
                    geometry = resolve_geometry_from_crib(result.geometry, width, height)
                    detection_method = result.method
                    
                    save_crib_config(
                        Path(args.config),
                        str(video_path),
                        result.geometry,
                        width,
                        height
                    )
                    print(f"[INFO] Saved crib configuration to {args.config}")
                else:
                    print(f"[WARN] Crib detection failed: {result.message}, falling back to config")
                    geometry = resolve_geometry(config, width, height)
        except Exception as e:
            print(f"[WARN] Error during crib detection: {e}, falling back to config")
            geometry = resolve_geometry(config, width, height)
    else:
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
    if "crib_geometry" in geometry:
        print("[GEOMETRY] Using polygon-based danger boundary")
    else:
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
                detection_method,
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
