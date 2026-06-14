"""
generate_web_demo_data.py

预计算YOLO检测结果，生成网页端Demo所需的数据文件
"""

import json
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

# 从demo_danger_action_detector导入相关函数
from demo_danger_action_detector import (
    load_config, load_yolo_model, run_detector, select_target,
    classify_stage, apply_debounce, get_trigger_x, resolve_geometry,
    Detection, CONFIG_PATH, DEFAULT_VIDEO_PATH
)


def generate_detection_data(
    video_path: Path,
    config_path: Path = CONFIG_PATH,
    output_dir: Path = Path("web_demo/data"),
) -> Dict[str, Any]:
    """
    对视频进行YOLO检测，生成每帧的检测结果数据
    """
    config = load_config(config_path)
    
    if not video_path.exists():
        print(f"[ERROR] Video file not found: {video_path}")
        return {}
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] Failed to open video file: {video_path}")
        return {}
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    ret, first_frame = cap.read()
    if not ret:
        print("[ERROR] Failed to read first frame.")
        cap.release()
        return {}
    
    geometry = resolve_geometry(config, width, height)
    model_name = config["model"].get("path", "yolo11n.pt")
    
    print(f"Loading YOLO model: {model_name}")
    model = load_yolo_model(model_name)
    
    detect_every_n = max(1, int(config["model"].get("detect_every_n_frames", 1)))
    warning_confirm_frames = int(config.get("debounce", {}).get("warning_confirm_frames", 3))
    danger_confirm_frames = int(config.get("debounce", {}).get("danger_confirm_frames", 5))
    lost_keep_frames = int(config.get("target_selection", {}).get("lost_keep_frames", 8))
    
    print(f"\nProcessing video: {video_path.name}")
    print(f"  Total frames: {total_frames}, FPS: {fps:.2f}, Resolution: {width}x{height}")
    print(f"  Detect every {detect_every_n} frames")
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    previous_target: Optional[Detection] = None
    current_target: Optional[Detection] = None
    lost_frames = 0
    warning_count = 0
    danger_count = 0
    displayed_stage = "SAFE"
    raw_stage = "SAFE"
    triggered = False
    
    frames_data = []
    events = []
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_idx += 1
        
        # 运行检测
        if frame_idx % detect_every_n == 0 or current_target is None:
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
        
        # 分类阶段
        raw_stage = classify_stage(current_target, geometry, config)
        displayed_stage, warning_count, danger_count = apply_debounce(
            raw_stage, displayed_stage, warning_count, danger_count,
            warning_confirm_frames, danger_confirm_frames,
        )
        
        if triggered:
            displayed_stage = "DANGEROUS_ACTION"
        
        # 记录触发事件
        if displayed_stage == "DANGEROUS_ACTION" and not triggered and current_target is not None:
            triggered = True
            events.append({
                "frame_index": frame_idx,
                "timestamp": round(frame_idx / fps, 2),
                "stage": "DANGEROUS_ACTION",
                "target_center": {"x": current_target.center[0], "y": current_target.center[1]},
                "target_bbox": {
                    "x": current_target.bbox[0],
                    "y": current_target.bbox[1],
                    "w": current_target.bbox[2],
                    "h": current_target.bbox[3],
                },
                "confidence": round(current_target.confidence, 4),
            })
        
        # 保存每一帧的数据（网页端需要逐帧平滑显示）
        if True:
            frame_data = {
                "frame_index": frame_idx,
                "timestamp": round(frame_idx / fps, 2),
                "stage": displayed_stage,
                "raw_stage": raw_stage,
                "warning_count": warning_count,
                "danger_count": danger_count,
            }
            
            if current_target is not None:
                frame_data["target"] = {
                    "center": {"x": current_target.center[0], "y": current_target.center[1]},
                    "bbox": {
                        "x": current_target.bbox[0],
                        "y": current_target.bbox[1],
                        "w": current_target.bbox[2],
                        "h": current_target.bbox[3],
                    },
                    "confidence": round(current_target.confidence, 4),
                    "class_name": current_target.class_name,
                }
            
            frames_data.append(frame_data)
        
        # 进度显示
        if frame_idx % 30 == 0:
            progress = (frame_idx / total_frames) * 100
            print(f"  Progress: {progress:.1f}% ({frame_idx}/{total_frames})", end="\r")
    
    cap.release()
    print(f"\n  Processing complete: {frame_idx} frames processed")
    
    # 构建输出数据
    output_data = {
        "video_info": {
            "filename": video_path.name,
            "total_frames": total_frames,
            "fps": round(fps, 2),
            "width": width,
            "height": height,
            "duration": round(total_frames / fps, 2),
        },
        "detection_config": {
            "model": model_name,
            "detect_every_n_frames": detect_every_n,
            "warning_confirm_frames": warning_confirm_frames,
            "danger_confirm_frames": danger_confirm_frames,
        },
        "geometry": {
            "safe_zone": geometry["safe_zone"],
            "warning_zone": geometry["warning_zone"],
            "danger_boundary_x": geometry["danger_boundary_x"],
        },
        "frames": frames_data,
        "events": events,
    }
    
    # 保存到文件
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"{video_path.stem}_detection.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"  Data saved to: {output_file}")
    print(f"  Total events: {len(events)}")
    
    return output_data


def main():
    parser = argparse.ArgumentParser(description="Generate web demo detection data")
    parser.add_argument("--video", type=str, default="data/dangerous_test1.mp4",
                        help="Path to input video")
    parser.add_argument("--config", type=str, default=str(CONFIG_PATH),
                        help="Path to detector config JSON")
    parser.add_argument("--output-dir", type=str, default="web_demo/data",
                        help="Output directory for detection data")
    args = parser.parse_args()
    
    video_path = Path(args.video)
    generate_detection_data(video_path, Path(args.config), Path(args.output_dir))


if __name__ == "__main__":
    main()
