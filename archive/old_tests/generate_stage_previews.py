"""生成各阶段屏幕反馈截图用于验证（ASCII文件名）"""
import sys
sys.path.insert(0, '.')

import cv2
import numpy as np
from demo_full_stage import BabyStage, StageClassifier, ScreenFeedbackRenderer

cap = cv2.VideoCapture('data/dangerous_test5.mp4')
ret, background = cap.read()
cap.release()

if not ret:
    print("Failed to read video")
    exit(1)

renderer = ScreenFeedbackRenderer()

# 阶段配置: (显示名, 文件名前缀)
stage_configs = [
    (BabyStage.DEEP_SLEEP,  "01_deep_sleep"),
    (BabyStage.WAKING,      "02_waking"),
    (BabyStage.CRYING_L1,   "03_crying_lv1"),
    (BabyStage.CRYING_L2,   "04_crying_lv2"),
    (BabyStage.CRYING_L3,   "05_crying_lv3"),
    (BabyStage.DANGER_2,    "06_danger_2"),
    (BabyStage.DANGER_3,    "07_danger_3"),
    (BabyStage.DANGER_4,    "08_danger_4"),
    (BabyStage.DANGER_5,    "09_danger_5"),
    (BabyStage.HAPPY_PLAY,  "10_happy_play"),
]

output_dir = 'output/stage_previews'
import os
os.makedirs(output_dir, exist_ok=True)

for i, (stage, prefix) in enumerate(stage_configs):
    frame = background.copy()

    # 渲染屏幕反馈
    renderer.render(frame, stage, None, fps=24.0)

    # 保存截图（ASCII文件名）
    filename = f'{output_dir}/{prefix}.png'
    ok = cv2.imwrite(filename, frame)
    print(f'  [{i+1:2d}/10] {stage:12s} -> {prefix}.png ... {"OK" if ok else "FAIL"}')

    # 刷新动画
    for _ in range(10):
        renderer.render(frame.copy(), stage, None, fps=24.0)

print()
print(f'Done! {len(stage_configs)} screenshots saved to {output_dir}/')
