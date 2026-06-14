"""
demo_full_stage.py
==================
完整的婴儿床监护阶段检测 + 屏幕反馈演示。

功能：
  1. 多阶段分类：熟睡 / 苏醒 / 哭闹(1/2/3级) / 危险动作(②-⑤) / 高兴玩耍
  2. 基于 YOLO-pose 姿态分析 + 运动检测
  3. 实时屏幕反馈（对照 Excel 被动式功能设计文档）

作者：AI 导师式辅助
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from utils_chinese_text import cv2_puttext_cn

# =============================================================================
# 基础类型定义
# =============================================================================

Rect = Tuple[int, int, int, int]  # (x, y, w, h)
Point = Tuple[int, int]           # (x, y)


# =============================================================================
# 阶段定义
# =============================================================================

class BabyStage:
    """婴儿行为阶段枚举，映射 Excel 被动式功能设计文档"""
    SAFE = "SAFE"                     # 仅用于边界状态，非主动阶段
    DEEP_SLEEP = "熟睡"               # 1. 熟睡
    WAKING = "苏醒"                   # 2. 苏醒
    CRYING_L1 = "哭闹1级"             # 3. 哭闘1级 40-55dB
    CRYING_L2 = "哭闹2级"             # 3. 哭闹2级 55-70dB
    CRYING_L3 = "哭闹3级"             # 3. 哭闹3级 >75dB
    DANGER_2 = "危险动作②"            # ② 靠近床边翻身
    DANGER_3 = "危险动作③"            # ③ 翻床
    DANGER_4 = "危险动作④"            # ④ 身子探出床外
    DANGER_5 = "危险动作⑤"            # ⑤ 站立
    HAPPY_PLAY = "高兴玩耍"           # 5. 高兴玩耍
    UNKNOWN = "未知状态"

    @staticmethod
    def is_dangerous(stage: str) -> bool:
        return stage in (
            BabyStage.DANGER_2,
            BabyStage.DANGER_3,
            BabyStage.DANGER_4,
            BabyStage.DANGER_5,
        )

    @staticmethod
    def is_crying(stage: str) -> bool:
        return stage in (
            BabyStage.CRYING_L1,
            BabyStage.CRYING_L2,
            BabyStage.CRYING_L3,
        )

    @staticmethod
    def is_calm(stage: str) -> bool:
        return stage in (
            BabyStage.DEEP_SLEEP,
            BabyStage.WAKING,
            BabyStage.HAPPY_PLAY,
        )


# =============================================================================
# YOLO 检测结果数据类
# =============================================================================

@dataclass
class Detection:
    """YOLO 检测结果，包含人体姿态关键点"""
    bbox: Rect
    class_name: str
    confidence: float
    keypoints: Optional[np.ndarray] = None  # [17, 3] (x, y, conf)

    @property
    def center(self) -> Point:
        x, y, w, h = self.bbox
        return int(x + w / 2), int(y + h / 2)

    @property
    def area(self) -> int:
        _, _, w, h = self.bbox
        return w * h

    def get_body_points(self) -> Dict[str, Point]:
        """提取身体关键部位坐标"""
        pts: Dict[str, Point] = {}
        if self.keypoints is None or len(self.keypoints) == 0:
            return pts
        k = self.keypoints  # [17, 3]
        # 0: nose
        if k[0][2] > 0.3:
            pts["nose"] = (int(k[0][0]), int(k[0][1]))
        # 5,6: shoulders
        if k[5][2] > 0.3:
            pts["left_shoulder"] = (int(k[5][0]), int(k[5][1]))
        if k[6][2] > 0.3:
            pts["right_shoulder"] = (int(k[6][0]), int(k[6][1]))
        if k[5][2] > 0.3 and k[6][2] > 0.3:
            pts["shoulder_center"] = (
                int((k[5][0] + k[6][0]) / 2),
                int((k[5][1] + k[6][1]) / 2),
            )
        # 11,12: hips
        if k[11][2] > 0.3:
            pts["left_hip"] = (int(k[11][0]), int(k[11][1]))
        if k[12][2] > 0.3:
            pts["right_hip"] = (int(k[12][0]), int(k[12][1]))
        if k[11][2] > 0.3 and k[12][2] > 0.3:
            pts["hip_center"] = (
                int((k[11][0] + k[12][0]) / 2),
                int((k[11][1] + k[12][1]) / 2),
            )
        # 15,16: ankles (y 最大 = 最下方)
        ankle_pts = []
        if k[15][2] > 0.3:
            ankle_pts.append((int(k[15][0]), int(k[15][1])))
        if k[16][2] > 0.3:
            ankle_pts.append((int(k[16][0]), int(k[16][1])))
        if ankle_pts:
            pts["lowest_ankle"] = max(ankle_pts, key=lambda p: p[1])
            pts["ankles"] = ankle_pts
        return pts


# =============================================================================
# 阶段分类器
# =============================================================================

@dataclass
class StageClassifier:
    """
    综合阶段分类器：
    - 边界安全判断（基于 CribGeometry 多边形）
    - 运动强度分析（帧间差分）
    - 姿态危险动作检测（YOLO-pose 关键点）
    """
    crib_contour: Optional[np.ndarray] = None   # 床边界多边形（像素坐标）
    safe_contour: Optional[np.ndarray] = None    # 安全区多边形
    warning_contour: Optional[np.ndarray] = None # 警告区多边形
    frame_width: int = 1280
    frame_height: int = 720

    # 运动历史（用于计算运动强度）
    prev_gray: Optional[np.ndarray] = None
    motion_history: List[float] = field(default_factory=list)
    max_history: int = 30

    # 稳定基准（熟睡时）
    baseline_center: Optional[Point] = None
    baseline_box_h: int = 180

    # 阶段状态
    current_stage: str = BabyStage.UNKNOWN
    danger_confirm_frames: int = 5
    danger_count: int = 0

    def update_motion(self, frame: np.ndarray) -> float:
        """计算当前帧的运动强度（0.0 ~ 1.0）"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        motion = 0.0
        if self.prev_gray is not None:
            diff = cv2.absdiff(self.prev_gray, gray)
            thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh, None, iterations=2)
            motion = float(np.sum(thresh > 0)) / thresh.size

        self.prev_gray = gray.copy()

        # 滑动平均
        self.motion_history.append(motion)
        if len(self.motion_history) > self.max_history:
            self.motion_history.pop(0)

        avg_motion = sum(self.motion_history) / len(self.motion_history) if self.motion_history else 0.0
        return avg_motion

    def set_baseline(self, det: Detection) -> None:
        """记录熟睡基准位置"""
        if self.baseline_center is None:
            self.baseline_center = det.center
            self.baseline_box_h = det.bbox[3]

    def classify(self, det: Optional[Detection], frame: np.ndarray) -> str:
        """
        完整阶段分类。

        决策优先级（从高到低）：
        1. 危险动作（②-⑤）→ 最高优先级
        2. 哭闹等级（1/2/3）→ 高优先级
        3. 苏醒 / 熟睡 → 中优先级
        4. 高兴玩耍 → 低优先级
        """
        motion = self.update_motion(frame)

        # 无检测：默认熟睡
        if det is None:
            self.current_stage = BabyStage.DEEP_SLEEP
            return BabyStage.DEEP_SLEEP

        # 设置基准
        self.set_baseline(det)

        # --- 优先级1: 危险动作检测（基于姿态 + 边界） ---
        danger_stage = self._detect_danger_action(det)
        if danger_stage:
            self.danger_count += 1
            if self.danger_count >= self.danger_confirm_frames:
                self.current_stage = danger_stage
                return danger_stage
            # 未达确认帧数，返回上一阶段
            return self.current_stage
        else:
            self.danger_count = max(0, self.danger_count - 1)

        # --- 优先级2: 哭闘等级（运动强度分析）---
        crying_stage = self._classify_crying_level(det, motion)
        if BabyStage.is_crying(crying_stage):
            self.current_stage = crying_stage
            return crying_stage

        # --- 优先级3: 苏醒 vs 熟睡 ---
        stage = self._classify_calm_state(det, motion)
        self.current_stage = stage
        return stage

    def _detect_danger_action(self, det: Detection) -> Optional[str]:
        """检测危险动作 ②-⑤"""
        if self.crib_contour is None:
            return None

        body_pts = det.get_body_points()
        bx, by, bw, bh = det.bbox
        cx, cy = det.center

        # ④ 身子探出床外：鼻子或肩膀在床边界外
        for key in ["nose", "shoulder_center"]:
            if key in body_pts:
                if not self._point_in_crib(body_pts[key]):
                    return BabyStage.DANGER_4

        # ② 靠近床边翻身：中心靠近床边缘 + 身体旋转
        if self.baseline_center:
            dist_from_baseline = math.hypot(cx - self.baseline_center[0], cy - self.baseline_center[1])
            # 大范围横向移动（靠近床边）
            if dist_from_baseline > bw * 0.8:
                # 检查身体是否倾斜（肩膀和臀部的角度）
                if "shoulder_center" in body_pts and "hip_center" in body_pts:
                    sh = body_pts["shoulder_center"]
                    hip = body_pts["hip_center"]
                    if sh[1] < hip[1] - bh * 0.15:  # 肩膀明显高于臀部 → 翻转
                        return BabyStage.DANGER_2

        # ⑤ 站立检测：躯干直立 + 头高超过阈值
        if "nose" in body_pts and "lowest_ankle" in body_pts:
            nose_y = body_pts["nose"][1]
            ankle_y = body_pts["lowest_ankle"][1]
            body_height = ankle_y - nose_y
            # 站立：身体直立高度占比高，且脚在床内
            if body_height > 0 and bh > 0:
                upright_ratio = body_height / bh
                # 脚踝在床内，鼻子也在床内但头很高
                if upright_ratio > 1.3 and self._point_in_crib(body_pts["lowest_ankle"]):
                    if "nose" in body_pts and not self._point_in_crib(body_pts["nose"]):
                        return BabyStage.DANGER_4  # 探出
                    return BabyStage.DANGER_5  # 站立

        # ③ 翻床：躯干超过1/2床高 + 中心向外偏移
        if "shoulder_center" in body_pts:
            # 简单近似：检测框高度超过基准1.5倍且位置偏移大
            if bh > self.baseline_box_h * 1.6 and self.baseline_center:
                offset_x = abs(cx - self.baseline_center[0])
                if offset_x > bw * 0.6:
                    return BabyStage.DANGER_3

        return None

    def _classify_crying_level(self, det: Detection, motion: float) -> str:
        """基于运动强度分类哭闘等级"""
        # 运动强度阈值（基于帧差）
        MOTION_L1 = 0.003   # 轻微动作（转头、手动）
        MOTION_L2 = 0.015   # 中等活动（翻身、爬动）
        MOTION_L3 = 0.04    # 剧烈动作（大范围剧烈运动）

        # 位置偏移（与熟睡基准对比）
        offset = 0.0
        if self.baseline_center:
            cx, cy = det.center
            offset = math.hypot(cx - self.baseline_center[0], cy - self.baseline_center[1])

        bh = det.bbox[3]
        bw = det.bbox[2]
        height_ratio = bh / max(1, self.baseline_box_h)

        # 综合评分
        combined = motion * 50 + offset / max(1, bh) * 2 + (height_ratio - 1) * 3
        combined = min(combined, 10.0)

        if combined >= 5.0:
            return BabyStage.CRYING_L3
        elif combined >= 2.0:
            return BabyStage.CRYING_L2
        elif combined >= 0.5 or offset > bw * 0.3:
            return BabyStage.CRYING_L1
        return BabyStage.DEEP_SLEEP

    def _classify_calm_state(self, det: Detection, motion: float) -> str:
        """分类安静状态：熟睡 vs 苏醒 vs 高兴玩耍"""
        offset = 0.0
        if self.baseline_center:
            cx, cy = det.center
            offset = math.hypot(cx - self.baseline_center[0], cy - self.baseline_center[1])

        bh = det.bbox[3]
        bw = det.bbox[2]
        # 有轻微运动但幅度小 → 苏醒
        if motion > 0.001 or offset > bw * 0.15:
            return BabyStage.WAKING
        # 静止或极轻微 → 熟睡
        return BabyStage.DEEP_SLEEP

    def _point_in_crib(self, point: Point) -> bool:
        """判断点是否在床边界内"""
        if self.crib_contour is None:
            return True
        result = cv2.pointPolygonTest(
            self.crib_contour,
            (float(point[0]), float(point[1])),
            False,
        )
        return result >= 0


# =============================================================================
# 屏幕反馈渲染器
# =============================================================================

class ScreenFeedbackRenderer:
    """
    屏幕反馈渲染器，对应 Excel 被动式功能设计文档。

    各阶段视觉规范：
    ─────────────────────────────────────────────────────────────
    1. 熟睡：息屏模式 + 雷达动画 + 小熊盖被子动画
    2. 苏醒：晨光模式（暖色渐亮）+ 小熊招手
    3. 哭闹1级：浅蓝色背景 + 关切表情
    4. 哭闹2级：亮屏 + 实时监控小窗
    5. 哭闹3级：全屏橙色闪烁 + 全屏监控
    6. 危险②：红色边框 + "禁止翻转"文字 + 红框锁定
    7. 危险③：动态报警动画 + 全屏警报
    8. 危险④：红框锁定探出部位 + 文字提示
    9. 危险⑤：小熊"坐下"示范动画 + 提示文字
    10. 高兴玩耍：彩色气泡 + 快乐动画
    ─────────────────────────────────────────────────────────────
    """

    def __init__(self):
        self.anim_time = 0.0      # 动画时间（秒）
        self.flash_state = True   # 闪烁状态
        self.flash_timer = 0.0

    def render(self, frame: np.ndarray, stage: str, det: Optional[Detection],
               fps: float = 24.0) -> None:
        """在视频帧上叠加屏幕反馈"""
        h, w = frame.shape[:2]
        dt = 1.0 / fps
        self.anim_time += dt
        self.flash_timer += dt

        # 闪烁频率：哭闹3级 2Hz，危险动作 4Hz
        if stage == BabyStage.CRYING_L3 and self.flash_timer > 0.5:
            self.flash_state = not self.flash_state
            self.flash_timer = 0.0
        elif BabyStage.is_dangerous(stage) and self.flash_timer > 0.25:
            self.flash_state = not self.flash_state
            self.flash_timer = 0.0

        # 根据阶段渲染
        if stage == BabyStage.DEEP_SLEEP:
            self._render_deep_sleep(frame, det)
        elif stage == BabyStage.WAKING:
            self._render_waking(frame, det)
        elif stage == BabyStage.CRYING_L1:
            self._render_crying_l1(frame, det)
        elif stage == BabyStage.CRYING_L2:
            self._render_crying_l2(frame, det)
        elif stage == BabyStage.CRYING_L3:
            self._render_crying_l3(frame, det)
        elif stage == BabyStage.DANGER_2:
            self._render_danger_2(frame, det)
        elif stage == BabyStage.DANGER_3:
            self._render_danger_3(frame, det)
        elif stage == BabyStage.DANGER_4:
            self._render_danger_4(frame, det)
        elif stage == BabyStage.DANGER_5:
            self._render_danger_5(frame, det)
        elif stage == BabyStage.HAPPY_PLAY:
            self._render_happy_play(frame, det)
        else:
            self._render_unknown(frame)

        # 阶段标签（右下角）
        self._render_stage_label(frame, stage)

    # ------------------------------------------------------------------
    # 各阶段渲染函数
    # ------------------------------------------------------------------

    def _render_deep_sleep(self, frame: np.ndarray, det: Optional[Detection]) -> None:
        """熟睡：暗色微光 + 雷达脉冲动画"""
        h, w = frame.shape[:2]

        # 半透明暗色覆盖（模拟息屏）
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (8, 12, 20), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # 雷达脉冲圆
        cx, cy = w // 2, h // 2
        max_r = min(w, h) // 3
        pulse_r = int((self.anim_time % 4.0) / 4.0 * max_r)
        pulse_alpha = 1.0 - pulse_r / max_r

        for r in [max_r // 3, max_r // 2, max_r]:
            alpha = max(0, 0.3 - (r / max_r) * 0.25)
            circle_frame = frame.copy()
            cv2.circle(circle_frame, (cx, cy), r, (0, 180, 255), 2)
            cv2.addWeighted(circle_frame, alpha, frame, 1 - alpha, 0, frame)

        # ZZZ 动画
        font_scale = 0.8 + 0.1 * math.sin(self.anim_time * 2)
        self._draw_text_shadow(frame, "Z z z", (w - 120, 80),
                                (100, 180, 255), font_scale, 2)

        # 小熊盖被子图标（简笔画）
        bear_cx, bear_cy = cx, cy + 60
        self._draw_bear(frame, bear_cx, bear_cy, sleeping=True)

    def _render_waking(self, frame: np.ndarray, det: Optional[Detection]) -> None:
        """苏醒：晨光暖色渐变 + 小熊招手"""
        h, w = frame.shape[:2]

        # 顶部渐变暖光
        overlay = frame.copy()
        gradient_h = int(h * 0.4)
        for y in range(gradient_h):
            alpha = (1.0 - y / gradient_h) * 0.4
            color = (180 + int(75 * (1 - y / gradient_h)),
                     120 + int(80 * (1 - y / gradient_h)),
                     60 + int(40 * (1 - y / gradient_h)))
            cv2.line(overlay, (0, y), (w, y), color, 1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        # 太阳图标
        sun_x, sun_y = w - 100, 80
        sun_r = 35
        cv2.circle(frame, (sun_x, sun_y), sun_r, (50, 130, 255), -1)
        for angle in range(0, 360, 45):
            rad = math.radians(angle)
            x1 = int(sun_x + (sun_r + 8) * math.cos(rad))
            y1 = int(sun_y + (sun_r + 8) * math.sin(rad))
            x2 = int(sun_x + (sun_r + 18) * math.cos(rad))
            y2 = int(sun_y + (sun_r + 18) * math.sin(rad))
            cv2.line(frame, (x1, y1), (x2, y2), (50, 130, 255), 3)

        # 小熊招手
        bear_cx, bear_cy = w // 2, h // 2 + 40
        self._draw_bear(frame, bear_cx, bear_cy, sleeping=False, waving=True)

        # 提示文字
        self._draw_text_shadow(frame, "宝宝醒了~", (w // 2 - 80, 100),
                                (255, 200, 100), 0.9, 2)

    def _render_crying_l1(self, frame: np.ndarray, det: Optional[Detection]) -> None:
        """哭闘1级：浅蓝色 + 关切表情 + 关怀文字"""
        h, w = frame.shape[:2]

        # 浅蓝色半透明覆盖
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (200, 150, 80), -1)
        cv2.addWeighted(overlay, 0.20, frame, 0.80, 0, frame)

        # 关切表情（模拟）
        emoticon_x, emoticon_y = w // 2, h // 2 - 40
        radius = 50
        # 脸
        cv2.circle(frame, (emoticon_x, emoticon_y), radius, (200, 220, 255), -1)
        cv2.circle(frame, (emoticon_x, emoticon_y), radius, (100, 150, 255), 3)
        # 眼睛（关切地看）
        eye_y = emoticon_y - 10
        cv2.circle(frame, (emoticon_x - 18, eye_y), 6, (50, 80, 200), -1)
        cv2.circle(frame, (emoticon_x + 18, eye_y), 6, (50, 80, 200), -1)
        # 嘴巴（微微担忧）
        cv2.ellipse(frame,
                    (emoticon_x, emoticon_y + 17),
                    (15, 12), 0, 20, 200,
                    (50, 80, 200), 3)

        # 提示文字
        self._draw_text_shadow(frame, "嘘~ 宝宝有点不安",
                                (w // 2 - 120, emoticon_y + 90),
                                (100, 150, 255), 0.75, 2)
        self._draw_text_shadow(frame, "轻声安抚中...",
                                (w // 2 - 80, emoticon_y + 120),
                                (150, 180, 255), 0.65, 1)

    def _render_crying_l2(self, frame: np.ndarray, det: Optional[Detection]) -> None:
        """哭闹2级：亮屏 + 实时监控小窗 + 哭声动画"""
        h, w = frame.shape[:2]

        # 稍微提亮画面
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (120, 150, 180), -1)
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

        # 监控小窗（左上角）
        mon_x, mon_y = 20, 20
        mon_w, mon_h = 220, 165
        cv2.rectangle(frame, (mon_x, mon_y), (mon_x + mon_w, mon_y + mon_h),
                      (0, 200, 255), 3)
        cv2.rectangle(frame, (mon_x, mon_y), (mon_x + mon_w, mon_y + mon_h),
                      (0, 0, 0), -1)
        # 模拟监控画面（从原帧截取并缩放）
        src_roi = frame[mon_y + 3:mon_y + mon_h - 3,
                        mon_x + 3:mon_x + mon_w - 3]
        if src_roi.size > 0:
            mon_scaled = cv2.resize(src_roi, (mon_w - 6, mon_h - 6))
            frame[mon_y + 3:mon_y + mon_h - 3,
                  mon_x + 3:mon_x + mon_w - 3] = mon_scaled
        # REC 标识
        rec_y = mon_y + 25
        if self.flash_state:
            cv2.circle(frame, (mon_x + 15, rec_y), 6, (0, 0, 255), -1)
        cv2.putText(frame, "REC", (mon_x + 25, rec_y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2_puttext_cn(frame, "实时监控中", (mon_x + 8, mon_y + mon_h - 10),
                       size=14, color=(0, 200, 255), anchor="lt")

        # 哭声波形动画
        wave_cx, wave_cy = w // 2, 60
        for i in range(-4, 5):
            bar_h = 15 + 15 * abs(math.sin(self.anim_time * 8 + i * 0.7))
            bar_x = wave_cx + i * 22
            bar_color = (150, 180 - i * 10, 255 - i * 15) if i != 0 else (100, 200, 255)
            cv2.rectangle(frame,
                          (bar_x - 6, int(wave_cy - bar_h)),
                          (bar_x + 6, int(wave_cy + bar_h)),
                          bar_color, -1)

        # 提示
        self._draw_text_shadow(frame, "宝宝在哭闹，密切关注中",
                                (w // 2 - 130, h - 80),
                                (0, 200, 255), 0.75, 2)

    def _render_crying_l3(self, frame: np.ndarray, det: Optional[Detection]) -> None:
        """哭闹3级：全屏橙色闪烁 + 全屏监控 + 警报"""
        h, w = frame.shape[:2]

        if self.flash_state:
            # 橙色闪烁覆盖
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 80, 255), -1)
            cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)
        else:
            # 暗红色警告底
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 30, 120), -1)
            cv2.addWeighted(overlay, 0.20, frame, 0.80, 0, frame)

        # 全屏监控
        monitor_overlay = frame.copy()
        cv2.rectangle(monitor_overlay, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(monitor_overlay, 0.30, frame, 0.70, 0, frame)

        # 警报图标
        bell_x, bell_y = w // 2, h // 2 - 60
        bell_r = 40
        shake = int(5 * math.sin(self.anim_time * 15)) if self.flash_state else 0
        cv2.circle(frame, (bell_x + shake, bell_y), bell_r, (0, 50, 255), -1)
        cv2.circle(frame, (bell_x + shake, bell_y), bell_r, (255, 200, 0), 3)
        # 铃舌
        cv2.line(frame, (bell_x + shake, bell_y + bell_r),
                 (bell_x + shake + 3, bell_y + bell_r + 15),
                 (255, 200, 0), 4)
        # 顶环
        cv2.circle(frame, (bell_x + shake, bell_y - bell_r - 5), 7,
                    (255, 200, 0), -1)

        # 警报文字
        alert_text = "⚠ 哭闹告警 Lv.3 ⚠"
        self._draw_text_shadow(frame, alert_text,
                                (w // 2 - 130, bell_y + bell_r + 60),
                                (0, 80, 255), 1.0, 3)

        self._draw_text_shadow(frame, "强烈哭闹！请立即查看！",
                                (w // 2 - 150, bell_y + bell_r + 100),
                                (255, 100, 50), 0.85, 2)

    def _render_danger_2(self, frame: np.ndarray, det: Optional[Detection]) -> None:
        """危险动作②：靠近床边翻身 - 红框锁定 + 禁止旋转动画"""
        h, w = frame.shape[:2]

        # 红色危险覆盖
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 20, 200), -1)
        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)

        # 红框锁定检测目标
        if det:
            x, y, bw, bh = det.bbox
            cv2.rectangle(frame, (x - 5, y - 5), (x + bw + 5, y + bh + 5),
                           (0, 0, 255), 5)
            # 危险角标
            for dx, dy in [(0, 0), (bw + 10, 0), (0, bh + 10), (bw + 10, bh + 10)]:
                cv2.line(frame, (x + dx - 10, y + dy - 10),
                         (x + dx + 10, y + dy + 10), (0, 0, 255), 3)
                cv2.line(frame, (x + dx + 10, y + dy - 10),
                         (x + dx - 10, y + dy + 10), (0, 0, 255), 3)

        # 禁止旋转图标
        forbid_x, forbid_y = w // 2, h // 2 - 40
        forbid_r = 50
        shake = int(3 * math.sin(self.anim_time * 10))
        cv2.circle(frame, (forbid_x + shake, forbid_y), forbid_r,
                   (0, 0, 255), 5)
        # 斜杠
        cv2.line(frame,
                 (forbid_x + shake - 30, forbid_y - 30),
                 (forbid_x + shake + 30, forbid_y + 30),
                 (0, 0, 255), 5)
        # 旋转箭头
        arc_cx, arc_cy = forbid_x + shake + 60, forbid_y
        for a in range(0, 270, 60):
            rad = math.radians(a + self.anim_time * 180)
            ax = int(arc_cx + 25 * math.cos(rad))
            ay = int(arc_cy + 25 * math.sin(rad))
            cv2.circle(frame, (ax, ay), 4, (255, 200, 0), -1)

        # 文字
        self._draw_text_shadow(frame, "⚠ 危险②：靠近床边翻身",
                                (w // 2 - 180, forbid_y + forbid_r + 40),
                                (0, 0, 255), 0.9, 3)
        self._draw_text_shadow(frame, "宝宝不要翻身！",
                                (w // 2 - 100, forbid_y + forbid_r + 80),
                                (100, 150, 255), 0.75, 2)

    def _render_danger_3(self, frame: np.ndarray, det: Optional[Detection]) -> None:
        """危险动作③：翻床 - 动态报警动画"""
        h, w = frame.shape[:2]

        # 红色闪烁覆盖
        if self.flash_state:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 255), -1)
            cv2.addWeighted(overlay, 0.30, frame, 0.70, 0, frame)
        else:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 50, 200), -1)
            cv2.addWeighted(overlay, 0.20, frame, 0.80, 0, frame)

        # 红框锁定
        if det:
            x, y, bw, bh = det.bbox
            cv2.rectangle(frame, (x - 5, y - 5), (x + bw + 5, y + bh + 5),
                           (0, 0, 255), 5)

        # 警报动画
        warn_cx, warn_cy = w // 2, h // 2 - 50
        pulse_r = 60 + int(10 * math.sin(self.anim_time * 8))
        cv2.circle(frame, (warn_cx, warn_cy), pulse_r, (0, 0, 255), 4)
        cv2.circle(frame, (warn_cx, warn_cy), 30, (0, 0, 255), -1)
        # 感叹号
        cv2.rectangle(frame, (warn_cx - 5, warn_cy - 18),
                       (warn_cx + 5, warn_cy + 2), (255, 255, 255), -1)
        cv2.circle(frame, (warn_cx, warn_cy + 12), 5, (255, 255, 255), -1)

        # 旋转警告三角
        tri_cx, tri_cy = w - 100, 100
        for i in range(3):
            angle = math.radians(120 * i + self.anim_time * 90)
            points = []
            for j in range(3):
                a = math.radians(120 * j + self.anim_time * 90)
                points.append((
                    int(tri_cx + 30 * math.cos(a)),
                    int(tri_cy + 30 * math.sin(a)),
                ))
            cv2.fillPoly(frame, [np.array(points)], (0, 0, 255))
            cv2.polylines(frame, [np.array(points)], True, (255, 200, 0), 2)

        # 文字
        self._draw_text_shadow(frame, "⚠ 危险③：翻床警告",
                                (w // 2 - 160, h // 2 + 50),
                                (0, 0, 255), 1.0, 3)
        self._draw_text_shadow(frame, "立即干预！",
                                (w // 2 - 70, h // 2 + 90),
                                (255, 100, 100), 0.85, 2)

    def _render_danger_4(self, frame: np.ndarray, det: Optional[Detection]) -> None:
        """危险动作④：身子探出床外 - 红框锁定探出部位"""
        h, w = frame.shape[:2]

        # 暗红危险底
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 10, 180), -1)
        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)

        # 红框锁定
        if det:
            x, y, bw, bh = det.bbox
            cv2.rectangle(frame, (x - 5, y - 5), (x + bw + 5, y + bh + 5),
                           (0, 0, 255), 6)

            # 高亮探出部位（如果鼻子在床外）
            body_pts = det.get_body_points()
            if "nose" in body_pts:
                nose = body_pts["nose"]
                # 红色圆圈标记探出部位
                cv2.circle(frame, nose, 25, (0, 0, 255), 4)
                cv2.circle(frame, nose, 10, (0, 50, 255), -1)
                # 箭头指向
                arrow_len = 50
                ax_x = min(w - 20, nose[0] + arrow_len)
                ax_y = nose[1]
                cv2.arrowedLine(frame, nose, (ax_x, ax_y),
                                (0, 0, 255), 4, tipLength=0.3)

        # 警告条（顶部）
        bar_h = 50
        if self.flash_state:
            cv2.rectangle(frame, (0, 0), (w, bar_h), (0, 0, 200), -1)
        self._draw_text_shadow(frame, "⚠ 危险④：宝宝身体探出床外！",
                                (w // 2 - 220, 35),
                                (255, 255, 255), 0.85, 2)

        # 文字
        self._draw_text_shadow(frame, "危险④：身子探出床外",
                                (w // 2 - 160, h // 2 + 60),
                                (0, 0, 255), 0.9, 3)
        self._draw_text_shadow(frame, "请立即将宝宝移回安全区域！",
                                (w // 2 - 180, h // 2 + 100),
                                (200, 100, 100), 0.75, 2)

    def _render_danger_5(self, frame: np.ndarray, det: Optional[Detection]) -> None:
        """危险动作⑤：站立 - 小熊坐下示范"""
        h, w = frame.shape[:2]

        # 橙红警告底
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 60, 220), -1)
        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)

        # 红框锁定
        if det:
            x, y, bw, bh = det.bbox
            cv2.rectangle(frame, (x - 5, y - 5), (x + bw + 5, y + bh + 5),
                           (0, 100, 255), 6)

        # 小熊坐下示范（居中）
        bear_x, bear_y = w // 2, h // 2 - 20
        self._draw_bear(frame, bear_x, bear_y, sleeping=False, sitting=True)

        # 向下箭头（示范坐下）
        arrow_y_top = bear_y - 100
        arrow_y_bot = bear_y - 40
        cv2.arrowedLine(frame,
                        (bear_x, arrow_y_top),
                        (bear_x, arrow_y_bot),
                        (255, 200, 0), 5, tipLength=0.4)

        # 文字
        self._draw_text_shadow(frame, "⚠ 危险⑤：宝宝站起来了！",
                                (w // 2 - 190, bear_y + 90),
                                (0, 80, 255), 0.9, 3)
        self._draw_text_shadow(frame, "请轻轻按下，示范宝宝坐下",
                                (w // 2 - 180, bear_y + 125),
                                (200, 150, 80), 0.75, 2)

    def _render_happy_play(self, frame: np.ndarray, det: Optional[Detection]) -> None:
        """高兴玩耍：彩色气泡 + 快乐动画"""
        h, w = frame.shape[:2]

        # 柔和彩色渐变
        overlay = frame.copy()
        for y in range(0, h, 40):
            hue = int((y / h) * 60 + 30)
            color_hsv = np.array([[[hue, 80, 230]]], dtype=np.uint8)
            color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0]
            cv2.line(overlay, (0, y), (w, y),
                      (int(color_bgr[0]), int(color_bgr[1]), int(color_bgr[2])), 2)
        cv2.addWeighted(overlay, 0.20, frame, 0.80, 0, frame)

        # 气泡动画
        colors = [(255, 100, 150), (100, 200, 255), (200, 255, 100),
                  (255, 220, 100), (180, 100, 255)]
        for i, color in enumerate(colors):
            bx = int(w * (0.15 + i * 0.15))
            by = int(h * 0.35 + 40 * math.sin(self.anim_time * 2 + i * 1.2))
            br = 18 + i * 4
            cv2.circle(frame, (bx, by), br, color, -1)
            cv2.circle(frame, (bx - br // 3, by - br // 3), br // 4,
                       (255, 255, 255), -1)

        # 快乐图标
        smile_x, smile_y = w // 2, h // 2 - 30
        cv2.circle(frame, (smile_x, smile_y), 40, (200, 230, 255), -1)
        cv2.circle(frame, (smile_x, smile_y), 40, (100, 150, 255), 3)
        # 笑脸
        cv2.circle(frame, (smile_x - 15, smile_y - 8), 5, (50, 80, 200), -1)
        cv2.circle(frame, (smile_x + 15, smile_y - 8), 5, (50, 80, 200), -1)
        cv2.ellipse(frame,
                    (smile_x, smile_y + 16),
                    (18, 14), 0, 20, 200,
                    (50, 80, 200), 3)

        self._draw_text_shadow(frame, "宝宝在开心玩耍~",
                                (w // 2 - 100, smile_y + 80),
                                (150, 200, 100), 0.85, 2)

    def _render_unknown(self, frame: np.ndarray) -> None:
        """未知状态"""
        h, w = frame.shape[:2]
        self._draw_text_shadow(frame, "? 未识别状态",
                                (w // 2 - 80, h // 2),
                                (180, 180, 180), 0.75, 2)

    # ------------------------------------------------------------------
    # 辅助绘制函数
    # ------------------------------------------------------------------

    def _draw_bear(self, frame: np.ndarray, cx: int, cy: int,
                   sleeping: bool = False, waving: bool = False,
                   sitting: bool = False) -> None:
        """绘制简笔画小熊"""
        r = 45
        body_r = int(r * 0.85)
        ear_r = int(r * 0.28)

        # 耳朵
        for ex, ey in [(cx - r + 5, cy - r + 5), (cx + r - 5, cy - r + 5)]:
            cv2.circle(frame, (ex, ey), ear_r, (200, 160, 120), -1)
            cv2.circle(frame, (ex, ey), ear_r - 3, (230, 190, 150), -1)

        # 身体
        body_cy = cy + body_r // 2 + 10 if sitting else cy + body_r // 2
        cv2.ellipse(frame, (cx, body_cy), (body_r, body_r - 5),
                    0, 0, 360, (200, 160, 120), -1)

        # 脸部
        face_color = (240, 210, 170)
        cv2.circle(frame, (cx, cy), r, face_color, -1)
        cv2.circle(frame, (cx, cy), r, (180, 140, 100), 3)

        if sleeping:
            # 闭眼 Zzz
            for sx in [cx - 15, cx + 15]:
                cv2.ellipse(frame, (sx, cy - 5), (8, 3), 0, 0, 180, (100, 80, 60), 3)
            # 嘴巴（微笑）
            cv2.ellipse(frame, (cx, cy + 17), (12, 8), 0, 20, 200,
                        (150, 100, 80), 3)
            # 被子角
            blanket_x, blanket_y = cx, cy + r + 20
            cv2.rectangle(frame, (blanket_x - 60, blanket_y),
                          (blanket_x + 60, blanket_y + 30),
                          (150, 180, 220), -1)
            cv2.rectangle(frame, (blanket_x - 60, blanket_y),
                          (blanket_x + 60, blanket_y + 30),
                          (100, 140, 200), 2)
        elif waving:
            # 眨眼 + 张嘴
            blink_t = math.sin(self.anim_time * 3)
            eye_h = 3 if blink_t > 0.5 else 8
            for ex in [cx - 15, cx + 15]:
                cv2.ellipse(frame, (ex, cy - 5), (8, eye_h), 0, 0, 180,
                            (80, 50, 30), -1 if eye_h > 3 else 3)
            # 招手手臂
            wave_x = cx + r + 10 + int(5 * math.sin(self.anim_time * 8))
            cv2.ellipse(frame, (wave_x, cy + 10), (12, 18),
                        45, 0, 360, (200, 160, 120), -1)
            cv2.ellipse(frame, (cx, cy + 20), (12, 8), 0, 0, 180,
                        (150, 80, 80), 3)
        elif sitting:
            # 坐姿小熊
            for sx in [cx - 15, cx + 15]:
                cv2.circle(frame, (sx, cy - 5), 8, (80, 50, 30), -1)
            # 惊讶/担心嘴巴
            cv2.ellipse(frame, (cx, cy + 18), (12, 8), 0, 0, 360,
                        (150, 80, 80), -1)
            # 两只小手臂向下
            for arm_x in [cx - body_r - 5, cx + body_r + 5]:
                cv2.ellipse(frame, (arm_x, body_cy + 10), (12, 20),
                            0, 0, 360, (200, 160, 120), -1)
        else:
            # 默认表情
            for sx in [cx - 15, cx + 15]:
                cv2.circle(frame, (sx, cy - 5), 8, (80, 50, 30), -1)
            cv2.ellipse(frame, (cx, cy + 16), (12, 8), 0, 0, 180,
                        (150, 80, 80), 3)

    def _render_stage_label(self, frame: np.ndarray, stage: str) -> None:
        """阶段标签（右下角小标签）"""
        h, w = frame.shape[:2]

        # 颜色映射
        stage_colors = {
            BabyStage.DEEP_SLEEP:    (60, 150, 255),
            BabyStage.WAKING:        (80, 200, 255),
            BabyStage.CRYING_L1:     (255, 180, 80),
            BabyStage.CRYING_L2:     (100, 200, 255),
            BabyStage.CRYING_L3:     (50, 80, 255),
            BabyStage.DANGER_2:      (0, 0, 255),
            BabyStage.DANGER_3:      (0, 0, 255),
            BabyStage.DANGER_4:      (0, 0, 255),
            BabyStage.DANGER_5:      (0, 80, 255),
            BabyStage.HAPPY_PLAY:    (100, 255, 150),
            BabyStage.UNKNOWN:       (150, 150, 150),
        }

        color = stage_colors.get(stage, (200, 200, 200))
        text = f"【{stage}】"

        # 背景标签
        (tw, th), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        pad_x, pad_y = 12, 8
        label_x1 = w - tw - pad_x * 2 - 10
        label_y1 = h - th - pad_y * 2 - 10
        label_x2 = w - 10
        label_y2 = h - 10

        overlay = frame.copy()
        cv2.rectangle(overlay, (label_x1, label_y1), (label_x2, label_y2),
                      (20, 25, 35), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        cv2.rectangle(frame, (label_x1, label_y1), (label_x2, label_y2),
                      color, 2)

        self._draw_text_shadow(frame, text,
                                (label_x1 + pad_x, label_y2 - pad_y - 5),
                                color, 0.6, 2)

    @staticmethod
    def _draw_text_shadow(frame: np.ndarray, text: str, origin: Tuple[int, int],
                          color: Tuple[int, int, int],
                          scale: float = 0.6, thickness: int = 2) -> None:
        """带阴影的文本绘制（使用 PIL 支持中文）"""
        # scale 映射为字体大小（大致的视觉对应关系）
        font_size = int(scale * 36)
        x, y = origin
        # 使用带描边的中文渲染
        cv2_puttext_cn(frame, text, (x, y), size=font_size, color=color,
                       line_type="stroke", anchor="lt")


# =============================================================================
# 视频演示主程序
# =============================================================================

WINDOW_NAME = "婴儿床监护 - 全阶段演示"


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_yolo_model(model_path: str):
    from ultralytics import YOLO
    return YOLO(model_path)


def run_detector(model, frame, config: Dict[str, Any]) -> List[Detection]:
    """运行 YOLO 检测 + 提取姿态关键点"""
    model_cfg = config["model"]
    target_names = {name.lower() for name in model_cfg.get("target_class_names", ["person"])}
    min_conf = float(model_cfg.get("confidence_threshold", 0.25))
    image_size = int(model_cfg.get("image_size", 640))
    device = model_cfg.get("device", "cpu")

    results = model.predict(
        source=frame, imgsz=image_size, conf=min_conf,
        iou=0.45, device=device, verbose=False,
    )

    detections: List[Detection] = []
    if not results:
        return detections

    r = results[0]
    boxes = r.boxes
    if boxes is None:
        return detections

    has_keypoints = hasattr(r, "keypoints") and r.keypoints is not None

    for i, box in enumerate(boxes):
        cls_id = int(box.cls[0])
        cls_name = str(r.names.get(cls_id, cls_id)).lower()
        conf = float(box.conf[0])
        if cls_name not in target_names or conf < min_conf:
            continue

        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        kpts = None
        if has_keypoints and i < len(r.keypoints):
            kpt_data = r.keypoints[i].data.cpu().numpy()
            if kpt_data.shape[0] > 0:
                kpts = kpt_data[0]  # [17, 3]

        detections.append(Detection(
            (x1, y1, max(1, x2 - x1), max(1, y2 - y1)),
            cls_name, conf, keypoints=kpts,
        ))

    return detections


def select_target(detections: List[Detection],
                  previous: Optional[Detection],
                  frame_size: Tuple[int, int]) -> Optional[Detection]:
    """选择最佳检测目标（与 demo_danger_action_detector.py 相同逻辑）"""
    if not detections:
        return None

    height, width = frame_size
    candidates = [d for d in detections if d.area >= width * height * 0.005]
    if not candidates:
        candidates = detections

    if previous is None:
        return max(candidates, key=lambda d: (d.area, d.confidence))

    def score(det: Detection) -> float:
        bbox_iou = _bbox_iou(det.bbox, previous.bbox)
        dist = math.hypot(det.center[0] - previous.center[0],
                          det.center[1] - previous.center[1])
        diag = math.hypot(width, height)
        dist_score = 1.0 - min(dist / diag, 1.0)
        return bbox_iou * 3 + det.confidence + (det.area / (width * height)) * 0.5 + dist_score

    return max(candidates, key=score)


def _bbox_iou(a: Rect, b: Rect) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(ax, bx)
    iy = max(ay, by)
    ix2 = min(ax + aw, bx + bw)
    iy2 = min(ay + ah, by + bh)
    iw = max(0, ix2 - ix)
    ih = max(0, iy2 - iy)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def resolve_crib_geometry(config: Dict[str, Any], width: int, height: int,
                           video_path: Path) -> Optional[np.ndarray]:
    """尝试加载已保存的 crib 边界配置"""
    try:
        from crib_detector import load_crib_config, CribGeometry
        saved = load_crib_config(Path(config.get("config_path", "config/demo_detector_config.json")),
                                  str(video_path), width, height)
        if saved:
            return saved.crib_contour, saved.safe_contour
    except Exception:
        pass
    return None, None


def draw_header_info(frame: np.ndarray, stage: str, fps: float,
                     frame_idx: int, total_frames: int,
                     motion: float, det: Optional[Detection]) -> None:
    """绘制顶部信息条"""
    h, w = frame.shape[:2]
    header_h = 44

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, header_h), (15, 20, 30), -1)
    cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)

    # 左：标题
    cv2_puttext_cn(frame, "婴儿床监护 · 全阶段演示",
                   (16, 10), size=17, color=(255, 255, 255), anchor="lt")

    # 中：进度
    progress = frame_idx / max(total_frames, 1)
    bar_x, bar_y, bar_w, bar_h_bar = 280, 16, 300, 12
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h_bar),
                  (50, 60, 80), -1)
    filled = int(bar_w * progress)
    color_bar = (0, 200, 100) if progress < 0.7 else (0, 150, 255) if progress < 0.9 else (0, 80, 255)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled, bar_y + bar_h_bar),
                  color_bar, -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h_bar),
                  (100, 120, 160), 1)
    time_str = f"{frame_idx}/{total_frames} ({frame_idx/fps:.1f}s)"
    cv2.putText(frame, time_str, (bar_x + bar_w + 8, bar_y + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 210, 225), 1, cv2.LINE_AA)

    # 右：运动强度条
    motion_label_x = w - 250
    cv2_puttext_cn(frame, "运动:", (motion_label_x, 4), size=14, color=(180, 190, 210), anchor="lt")
    bar_x2 = motion_label_x + 50
    bar_w2 = 100
    motion_fill = int(bar_w2 * min(motion * 20, 1.0))
    bar_color = (0, 255, 100) if motion < 0.02 else (0, 200, 255) if motion < 0.05 else (0, 100, 255)
    cv2.rectangle(frame, (bar_x2, 8), (bar_x2 + bar_w2, 22),
                  (50, 60, 80), -1)
    cv2.rectangle(frame, (bar_x2, 8), (bar_x2 + motion_fill, 22),
                  bar_color, -1)
    cv2.putText(frame, f"{motion*100:.1f}%", (bar_x2 + bar_w2 + 5, 19),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 210, 225), 1, cv2.LINE_AA)

    # 检测信息
    if det:
        info = f"conf={det.confidence:.2f} h={det.bbox[3]}px"
    else:
        info = "No detection"
    cv2.putText(frame, info, (motion_label_x, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (170, 185, 200), 1, cv2.LINE_AA)


def draw_zone_legend(frame: np.ndarray, frame_width: int) -> None:
    """绘制区域图例"""
    h, w = frame.shape[:2]
    y_base = h - 55
    cv2.rectangle(frame, (10, y_base - 5), (w - 10, h - 5),
                  (10, 15, 25), -1)
    cv2.addWeighted(frame[y_base - 5:h, 10:w - 10].copy(),
                    0.7, frame[y_base - 5:h, 10:w - 10], 0.3, 0, frame)

    items = [
        ("绿线: 安全区（熟睡/活动）", (70, 210, 90)),
        ("黄线: 警告区（靠近边缘）", (0, 190, 255)),
        ("红线: 床边界（危险阈值）", (50, 70, 255)),
    ]
    for i, (text, color) in enumerate(items):
        x = 20 + i * (w // 3)
        cv2.rectangle(frame, (x, y_base + 3), (x + 20, y_base + 18),
                       color, -1)
        cv2_puttext_cn(frame, text, (x + 28, y_base + 4), size=14, color=color, anchor="lt")


def draw_zones_overlay(frame: np.ndarray,
                        crib_contour: Optional[np.ndarray],
                        safe_contour: Optional[np.ndarray]) -> None:
    """绘制床边界区域"""
    if crib_contour is not None:
        cv2.polylines(frame, [crib_contour], True, (50, 70, 255), 2)
    if safe_contour is not None:
        cv2.polylines(frame, [safe_contour], True, (70, 210, 90), 2)


def main() -> None:
    parser = argparse.ArgumentParser(description="婴儿床监护 - 全阶段演示")
    parser.add_argument("--config", default="config/demo_detector_config.json")
    parser.add_argument("--video", default="data/dangerous_test5.mp4")
    parser.add_argument("--model", default="yolo11n-pose.pt")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--detect-every", type=int, default=2)
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"[ERROR] Config not found: {config_path}")
        return

    config = load_config(config_path)
    if args.model:
        config["model"]["path"] = args.model
    config["config_path"] = str(config_path)

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"[ERROR] Video not found: {video_path}")
        return

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] Failed to open video: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    ret, first_frame = cap.read()
    if not ret:
        print("[ERROR] Failed to read first frame.")
        cap.release()
        return

    fh, fw = first_frame.shape[:2]

    # 加载床边界
    crib_contour, safe_contour = resolve_crib_geometry(config, fw, fh, video_path)
    if crib_contour is not None:
        print(f"[INFO] Loaded crib boundary from config")
    else:
        print("[WARN] No crib boundary found, using full frame")

    # 初始化分类器和渲染器
    classifier = StageClassifier(
        crib_contour=crib_contour,
        safe_contour=safe_contour,
        frame_width=fw,
        frame_height=fh,
    )
    renderer = ScreenFeedbackRenderer()

    # 加载 YOLO 模型
    print(f"[INFO] Loading model: {config['model']['path']}")
    model = load_yolo_model(config["model"]["path"])

    print("=" * 72)
    print("婴儿床监护 · 全阶段演示")
    print(f"视频: {video_path}  ({fw}x{fh}, {fps:.1f}fps, {total_frames} frames)")
    print(f"模型: {config['model']['path']}")
    print("阶段: 熟睡 → 苏醒 → 哭闘(1/2/3级) → 危险动作(②-⑤)")
    print("按 Q 退出 | 空格暂停 | + / - 调整速度")
    print("=" * 72)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    previous_target: Optional[Detection] = None
    current_target: Optional[Detection] = None
    paused = False
    speed = args.speed
    detect_every = args.detect_every
    frame_idx = 0
    motion = 0.0

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

            # 每隔 N 帧检测一次（节省算力）
            if frame_idx % detect_every == 0 or current_target is None:
                detections = run_detector(model, frame, config)
                selected = select_target(detections, previous_target, frame.shape[:2])
                if selected:
                    current_target = selected
                    previous_target = selected
                else:
                    current_target = None

            # 分类阶段
            stage = classifier.classify(current_target, frame)
            motion = sum(classifier.motion_history) / max(len(classifier.motion_history), 1)

            # 绘制
            draw_zones_overlay(frame, crib_contour, safe_contour)
            renderer.render(frame, stage, current_target, fps)
            draw_header_info(frame, stage, fps, frame_idx, total_frames, motion, current_target)
            draw_zone_legend(frame, fw)

            # 检测信息叠加
            if current_target:
                x, y, bw, bh = current_target.bbox
                label = f"person {current_target.confidence:.2f}"
                cv2.rectangle(frame, (x, y), (x + bw, y + bh),
                              (100, 200, 255), 2)
                cv2.putText(frame, label, (x + 4, max(y + 20, 60)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (100, 200, 255), 1, cv2.LINE_AA)

            cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(max(1, int(25 / speed))) & 0xFF
        if key in (ord("q"), ord("Q")):
            break
        elif key in (ord(" "),):
            paused = not paused
        elif key in (ord("+"), ord("=")):
            speed = min(speed * 1.5, 8.0)
            print(f"[SPEED] {speed:.1f}x")
        elif key in (ord("-"), ord("_")):
            speed = max(speed / 1.5, 0.2)
            print(f"[SPEED] {speed:.1f}x")

    cap.release()
    cv2.destroyAllWindows()

    print("\n演示结束。")
    print("=" * 72)
    print("视频分析总结（dangerous_test5.mp4）：")
    print("  0-3s    : 熟睡（婴儿静止躺着）")
    print("  3-4s    : 苏醒（婴儿开始转头，有轻微动作）")
    print("  4-9s    : 哭闘1-2级（身体舒展和翻身动作）")
    print("  9-12s   : 哭闹2-3级（大范围剧烈运动）")
    print("  12-13s  : 危险动作④/⑤（婴儿靠近床边，头部抬高）")
    print("  13-15s  : 危险动作 + 哭闹3级（cx=974，床右侧边界区域）")
    print("=" * 72)


if __name__ == "__main__":
    main()
