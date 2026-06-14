"""
验证脚本：保存关键帧 + 屏幕反馈截图，验证全阶段演示效果。
在以下时间点截图：
  f24  (1s)   : 熟睡阶段
  f84  (3.5s) : 苏醒阶段
  f144 (6s)   : 哭闹1级
  f240 (10s)  : 哭闹2级
  f336 (14s)  : 哭闘3级
"""

import sys
sys.path.insert(0, ".")

import cv2
import numpy as np
from pathlib import Path

from demo_full_stage import (
    BabyStage, StageClassifier, ScreenFeedbackRenderer,
    resolve_crib_geometry, load_config, load_yolo_model,
    run_detector, select_target, draw_zones_overlay,
    draw_header_info, draw_zone_legend, WINDOW_NAME,
)


def save_key_frames():
    config_path = Path("config/demo_detector_config.json")
    video_path = Path("data/dangerous_test5.mp4")
    output_dir = Path("output/demo_frames")
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(config_path)
    config["config_path"] = str(config_path.resolve())
    config["model"]["path"] = "yolo11n-pose.pt"

    # 加载视频
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video: {fw}x{fh}, {fps:.1f}fps, {total_frames} frames")

    # 加载床边界
    crib_contour, safe_contour = resolve_crib_geometry(config, fw, fh, video_path)
    if crib_contour is not None:
        print(f"[OK] Crib boundary loaded: {crib_contour.shape}")
    else:
        print("[WARN] No crib boundary, using None")

    # 加载模型
    print(f"[INFO] Loading model: {config['model']['path']} ...")
    model = load_yolo_model(config["model"]["path"])

    # 初始化
    classifier = StageClassifier(
        crib_contour=crib_contour,
        safe_contour=safe_contour,
        frame_width=fw,
        frame_height=fh,
    )
    renderer = ScreenFeedbackRenderer()

    # 关键帧配置: (frame_idx, label, description)
    key_frames = [
        (24,  "01_deep_sleep",  "熟睡阶段 (1s)"),
        (84,  "02_waking",      "苏醒阶段 (3.5s)"),
        (144, "03_crying_lv1",  "哭闹1级 (6s)"),
        (240, "04_crying_lv2",  "哭闹2级 (10s)"),
        (336, "05_crying_lv3",  "哭闹3级 (14s)"),
    ]
    key_dict = {f[0]: (f[1], f[2]) for f in key_frames}  # frame_idx -> (label, desc)

    prev_target = None
    curr_target = None
    frame_idx = 0
    saved_count = 0

    print("\n开始逐帧处理...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        # 每隔3帧检测一次（节省算力）
        if frame_idx % 3 == 0 or curr_target is None:
            detections = run_detector(model, frame, config)
            selected = select_target(detections, prev_target, frame.shape[:2])
            if selected:
                curr_target = selected
                prev_target = selected
            else:
                curr_target = None

        # 分类
        stage = classifier.classify(curr_target, frame)
        motion = sum(classifier.motion_history) / max(len(classifier.motion_history), 1)

        # 保存关键帧
        if frame_idx in key_dict:
            label, desc = key_dict[frame_idx]
            # 深拷贝避免影响原帧
            canvas = frame.copy()
            draw_zones_overlay(canvas, crib_contour, safe_contour)
            renderer.render(canvas, stage, curr_target, fps)
            draw_header_info(canvas, stage, fps, frame_idx, total_frames, motion, curr_target)
            draw_zone_legend(canvas, fw)

            if curr_target:
                x, y, bw, bh = curr_target.bbox
                cv2.rectangle(canvas, (x, y), (x + bw, y + bh), (100, 200, 255), 2)
                cv2.putText(canvas, f"person {curr_target.confidence:.2f}",
                           (x + 4, max(y + 20, 60)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                           (100, 200, 255), 1, cv2.LINE_AA)

            fname = f"{frame_idx:04d}_{label}.png"
            out_path = output_dir / fname
            cv2.imwrite(str(out_path), canvas)
            print(f"  [SAVED] f{frame_idx:3d} -> {fname}  "
                  f"| stage={stage} | motion={motion*100:.1f}%")

            if curr_target:
                cx, cy = curr_target.center
                print(f"           bbox_center=({cx},{cy}) h={curr_target.bbox[3]}px")

            saved_count += 1

        # 提前退出（已保存所有关键帧）
        if saved_count >= len(key_frames) and frame_idx > max(key_dict.keys()) + 5:
            break

    cap.release()
    print(f"\n完成！已保存 {saved_count} 张关键帧截图到 {output_dir}/")
    print("\n阶段分类总结：")
    for frame_n, label, desc in key_frames:
        print(f"  f{frame_n:3d} ({desc}): {label}")

    return output_dir


if __name__ == "__main__":
    save_key_frames()
