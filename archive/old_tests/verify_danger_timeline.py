"""
危险动作全帧扫描 (v2)：使用 demo_full_stage.py 原生管道，
分析 dangerous_test5.mp4 的全部帧，找出所有危险动作触发帧。
"""
import sys
sys.path.insert(0, ".")

import cv2
import numpy as np
import json
from pathlib import Path

from demo_full_stage import (
    StageClassifier, BabyStage,
    resolve_crib_geometry, load_config, load_yolo_model,
    run_detector, select_target,
)


def run_full_analysis():
    # ---------- 初始化 ----------
    config_path = Path("config/demo_detector_config.json")
    video_path = Path("data/dangerous_test5.mp4")
    output_dir = Path("output/demo_frames/danger_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(config_path)
    from ultralytics import YOLO
    model = YOLO("yolo11n-pose.pt")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] 无法打开视频: {video_path}")
        return

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] 视频: {video_path}  ({width}x{height}, {fps}fps, {total} frames)")
    print("=" * 70)

    # 加载婴儿床边界
    crib_contour, safe_contour = resolve_crib_geometry(
        config, width, height, str(video_path)
    )
    if crib_contour is not None:
        print(f"[OK] 婴儿床边界已加载: {crib_contour.shape}")
        classifier = StageClassifier(crib_contour=crib_contour)
    else:
        print("[WARN] 未加载到婴儿床边界")
        classifier = StageClassifier()

    # ---------- 全帧扫描 ----------
    records = []
    prev_gray = None
    prev_det = None   # 跟踪上一帧的检测结果

    for frame_idx in range(total):
        ret, frame = cap.read()
        if not ret:
            break

        # 运行检测
        results = run_detector(model, frame, config)
        det = select_target(results, prev_det, (height, width))
        if det is not None:
            prev_det = det

        # 使用原生分类器
        stage = classifier.classify(det, frame)

        # 提取信息
        record = {
            "frame": frame_idx,
            "t_s": round(frame_idx / fps, 2),
            "stage": str(stage),
            "is_danger": stage in [BabyStage.DANGER_2, BabyStage.DANGER_3, BabyStage.DANGER_4, BabyStage.DANGER_5],
            "danger_type": str(stage) if stage and str(stage).startswith("危险动作") else None,
            "has_detection": det is not None,
            "bbox": None,
            "center": None,
            "bbox_h": 0,
            "motion": round(sum(classifier.motion_history[-5:]) / max(len(classifier.motion_history[-5:]), 1) * 100, 2) if classifier.motion_history else 0,
        }

        if det is not None:
            bx, by, bw, bh = det.bbox
            record["bbox"] = (bx, by, bw, bh)
            record["center"] = det.center
            record["bbox_h"] = int(bh)

        records.append(record)

        # 进度
        if (frame_idx + 1) % 50 == 0 or frame_idx == total - 1:
            print(f"  进度: {frame_idx+1}/{total}  ({(frame_idx+1)/total*100:.1f}%)")

    cap.release()

    # ---------- 分析 ----------
    print("\n" + "=" * 70)
    print("全帧分析完成！")
    print("=" * 70)

    # 阶段分布
    stage_counts = {}
    for r in records:
        s = r["stage"]
        stage_counts[s] = stage_counts.get(s, 0) + 1

    print("\n[阶段分布]")
    for s, cnt in sorted(stage_counts.items(), key=lambda x: -x[1]):
        print(f"  {s:24s}: {cnt:4d} 帧  ({cnt/total*100:.1f}%)")

    # 危险帧
    danger_records = [r for r in records if r["is_danger"]]
    print(f"\n[危险动作帧] 共 {len(danger_records)} 帧 / {total} 帧 ({len(danger_records)/total*100:.1f}%)")

    # 按类型分组
    danger_by_type = {}
    for r in danger_records:
        dt = r["danger_type"]
        if dt not in danger_by_type:
            danger_by_type[dt] = []
        danger_by_type[dt].append(r)

    danger_timeline = []
    for dt, frames in danger_by_type.items():
        frames = sorted(frames, key=lambda x: x["frame"])
        # 合并连续段（间隔<=3帧视为连续）
        segments = []
        start = frames[0]["frame"]
        prev = frames[0]["frame"]
        for f in frames[1:]:
            if f["frame"] - prev > 3:
                if prev - start >= 4:  # 至少4帧才算有效段
                    segments.append((start, prev))
                start = f["frame"]
            prev = f["frame"]
        if prev - start >= 4:
            segments.append((start, prev))

        danger_timeline.append({
            "type": dt,
            "total_frames": len(frames),
            "segments": segments,
        })

        print(f"\n  [{dt}] 共 {len(frames)} 帧")
        for seg in segments:
            t_start = seg[0] / fps
            t_end   = seg[1] / fps
            dur = seg[1] - seg[0] + 1
            print(f"    f{seg[0]:04d} ~ f{seg[1]:04d}  ({t_start:.1f}s ~ {t_end:.1f}s)  持续 {dur} 帧 / {dur/fps:.1f}s")

    # ---------- 保存危险帧截图 ----------
    print("\n[保存危险帧截图]")
    cap2 = cv2.VideoCapture(str(video_path))
    saved = 0

    for dt, frames in danger_by_type.items():
        frames = sorted(frames, key=lambda x: x["frame"])
        # 每段保存：首帧、中间帧、末帧
        idxs = []
        if len(frames) >= 3:
            idxs = [0, len(frames)//2, -1]
        elif len(frames) == 2:
            idxs = [0, -1]
        else:
            idxs = [0]

        # 去重（同帧不同段）
        seen = set()
        for i in idxs:
            r = frames[i]
            if r["frame"] in seen:
                continue
            seen.add(r["frame"])

            cap2.set(cv2.CAP_PROP_POS_FRAMES, r["frame"])
            ret, frame = cap2.read()
            if not ret:
                continue

            # 绘制婴儿床边界
            if crib_contour is not None:
                cv2.drawContours(frame, [crib_contour.astype(np.int32)], 0, (0, 255, 0), 2)
            if safe_contour is not None:
                cv2.drawContours(frame, [safe_contour.astype(np.int32)], 0, (255, 255, 0), 1)

            # 绘制 bbox
            if r["bbox"] is not None:
                bx, by, bw, bh = [int(v) for v in r["bbox"]]
                cv2.rectangle(frame, (bx, by), (bx+bw, by+bh), (0, 255, 0), 2)

                # 绘制中心点
                if r["center"] is not None:
                    cx, cy = int(r["center"][0]), int(r["center"][1])
                    cv2.circle(frame, (cx, cy), 6, (0, 255, 255), -1)

            # 危险标签
            label = f"DANGER: {dt} | f{r['frame']} | h={r['bbox_h']}px | m={r['motion']}%"
            cv2.putText(frame, label, (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            ts = r["frame"] / fps
            filename = f"danger_{dt.replace('危险动作','d')}_f{r['frame']:04d}_{ts:.2f}s.png"
            outpath = output_dir / filename
            cv2.imwrite(str(outpath), frame)
            print(f"  [保存] f{r['frame']} ({ts:.2f}s) -> {filename}")
            saved += 1

    cap2.release()

    # ---------- 关键时间节点 ----------
    print("\n" + "=" * 70)
    print("【关键时间节点】")
    print("=" * 70)
    prev_stage = None
    stage_transitions = []
    for r in records:
        if r["stage"] != prev_stage:
            stage_transitions.append(r)
            prev_stage = r["stage"]

    for st in stage_transitions:
        t = st["frame"] / fps
        marker = " ***" if st["is_danger"] else ""
        print(f"  f{st['frame']:4d} ({t:5.1f}s) -> 【{st['stage']}】{marker}")

    # 输出非危险帧中的检测信息（帮助理解为什么危险没触发）
    print("\n[非危险帧抽样（每20帧）]")
    for r in records[::20]:
        if r["has_detection"] and not r["is_danger"]:
            print(f"  f{r['frame']:4d} ({r['t_s']:5.1f}s)  stage={r['stage']:20s}  "
                  f"h={r['bbox_h']}px  center=({r['center'][0]:.0f},{r['center'][1]:.0f})")

    # ---------- 保存报告 ----------
    report = {
        "video": str(video_path),
        "total_frames": total,
        "fps": fps,
        "duration_s": round(total / fps, 2),
        "stage_distribution": stage_counts,
        "danger_summary": {
            "total_danger_frames": len(danger_records),
            "percentage": round(len(danger_records) / total * 100, 2),
            "by_type": {dt: len(frames) for dt, frames in danger_by_type.items()},
        },
        "danger_timeline": [
            {
                "type": item["type"],
                "total_frames": item["total_frames"],
                "segments": [
                    {"f_start": s[0], "f_end": s[1],
                     "t_start_s": round(s[0]/fps, 1),
                     "t_end_s": round(s[1]/fps, 1)}
                    for s in item["segments"]
                ],
            }
            for item in danger_timeline
        ],
        "stage_transitions": [
            {"frame": r["frame"], "t_s": round(r["frame"]/fps, 1), "stage": r["stage"]}
            for r in stage_transitions
        ],
        "saved_screenshots": saved,
    }

    report_path = output_dir / "danger_timeline_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n[完成] 报告: {report_path}")
    print(f"[完成] 截图: {saved} 张 -> {output_dir}/")

    return report


if __name__ == "__main__":
    run_full_analysis()
