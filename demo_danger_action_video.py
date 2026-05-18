"""
demo_danger_action_video.py

危险动作完整AI闭环Demo - 视频触发版

作用：
1. 播放 data/dangerous_test_video.mp4
2. 根据视频时间段模拟 safe / warning / dangerous_action 状态
3. 在 dangerous_action 阶段自动触发 demo_danger_action.py 中已经跑通的 LangGraph 闭环
4. 在画面上叠加安全区、预警区、危险边界和当前状态
5. 适合录屏展示

运行：
python demo_danger_action_video.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2

from demo_danger_action import build_danger_action_graph, create_mock_danger_event


VIDEO_PATH = Path("data/dangerous_test_video.mp4")


def draw_status_overlay(frame, stage: str, progress: float, triggered: bool):
    """
    在视频画面上叠加状态信息。
    """

    h, w = frame.shape[:2]

    # 安全区：画面中间偏内区域
    safe_left = int(w * 0.16)
    safe_top = int(h * 0.18)
    safe_right = int(w * 0.84)
    safe_bottom = int(h * 0.86)

    # 危险边界线：右侧床沿方向
    danger_x = int(w * 0.78)

    # 根据状态设置颜色
    if stage == "safe":
        status_text = "SAFE / 安全"
        color = (0, 180, 0)
        point_x = int(w * 0.48)
    elif stage == "warning":
        status_text = "WARNING / 靠近床沿"
        color = (0, 200, 255)
        point_x = int(w * 0.68)
    else:
        status_text = "DANGEROUS_ACTION / 危险动作"
        color = (0, 0, 255)
        point_x = int(w * 0.82)

    point_y = int(h * 0.52)

    # 安全区域框
    cv2.rectangle(
        frame,
        (safe_left, safe_top),
        (safe_right, safe_bottom),
        (0, 180, 0),
        2,
    )

    # 危险边界线
    cv2.line(
        frame,
        (danger_x, safe_top),
        (danger_x, safe_bottom),
        (0, 0, 255),
        3,
    )

    # 模拟宝宝/目标中心点
    cv2.circle(frame, (point_x, point_y), 12, color, -1)
    cv2.circle(frame, (point_x, point_y), 18, color, 2)

    # 顶部状态条
    overlay_h = 88
    cv2.rectangle(frame, (0, 0), (w, overlay_h), (0, 0, 0), -1)

    cv2.putText(
        frame,
        f"AI Crib Safety Demo | {status_text}",
        (24, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        color,
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        f"Video progress: {progress * 100:.1f}% | Closed-loop triggered: {triggered}",
        (24, 68),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (230, 230, 230),
        2,
        cv2.LINE_AA,
    )

    # 区域说明
    cv2.putText(
        frame,
        "Green box: safe area",
        (safe_left, max(24, safe_top - 12)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (0, 180, 0),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        "Red line: danger boundary",
        (max(20, danger_x - 220), safe_bottom + 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )

    return frame


def get_stage(progress: float) -> str:
    """
    根据视频播放进度模拟状态阶段。
    """
    if progress < 0.35:
        return "safe"
    if progress < 0.70:
        return "warning"
    return "dangerous_action"


def trigger_danger_closed_loop():
    """
    触发已经跑通的危险动作 LangGraph 闭环。
    """
    print("\n" + "=" * 70)
    print(">>> 视频检测到危险阶段，触发 dangerous_action AI闭环")
    print("=" * 70)

    event = create_mock_danger_event()
    event["source"] = "video_demo"
    event["description"] = "视频检测到婴儿靠近床沿并出现探身风险，触发危险动作闭环"

    app = build_danger_action_graph()
    final_state = app.invoke(event)

    print("\n>>> 视频触发闭环结果摘要")
    summary = {
        "event_id": final_state.get("event_id"),
        "state_id": final_state.get("state_id"),
        "risk_level": final_state.get("risk_level"),
        "decision_priority": final_state.get("decision", {}).get("priority"),
        "voice_text": final_state.get("voice_command", {}).get("text"),
        "notification_title": final_state.get("notification_command", {}).get("title"),
        "event_log_file": "logs/danger_action_events.jsonl",
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("=" * 70 + "\n")


def main():
    if not VIDEO_PATH.exists():
        print(f"[ERROR] 找不到视频文件：{VIDEO_PATH}")
        print("请确认视频路径为：data/dangerous_test_video.mp4")
        return

    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        print(f"[ERROR] 无法打开视频文件：{VIDEO_PATH}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 25

    delay = max(1, int(1000 / fps))

    print("=" * 70)
    print("危险动作完整AI闭环Demo - 视频触发版")
    print(f"视频文件：{VIDEO_PATH}")
    print(f"总帧数：{total_frames}")
    print(f"FPS：{fps:.2f}")
    print("按 Q 退出")
    print("=" * 70)

    triggered = False
    last_stage = None

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        progress = current_frame / total_frames if total_frames > 0 else 0

        stage = get_stage(progress)

        if stage != last_stage:
            print(f"[视频状态] progress={progress:.2f}, stage={stage}")
            last_stage = stage

        # 第一次进入危险阶段时触发闭环
        if stage == "dangerous_action" and not triggered:
            triggered = True
            trigger_danger_closed_loop()

        frame = draw_status_overlay(frame, stage, progress, triggered)

        cv2.imshow("Dangerous Action AI Closed-loop Demo", frame)

        key = cv2.waitKey(delay) & 0xFF
        if key == ord("q") or key == ord("Q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    print("\nDemo播放结束。")
    if triggered:
        print("[SUCCESS] 已完成：视频危险阶段 → dangerous_action → AI闭环触发。")
    else:
        print("[WARN] 视频播放结束，但未触发 dangerous_action。")


if __name__ == "__main__":
    main()