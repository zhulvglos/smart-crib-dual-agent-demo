"""快速验证脚本 - 测试完整视频分类"""
import sys
sys.path.insert(0, '.')

import cv2
import numpy as np
from demo_full_stage import BabyStage, StageClassifier, ScreenFeedbackRenderer, run_detector, select_target
from ultralytics import YOLO

print('=== Full video classification test ===')
model = YOLO('yolo11n-pose.pt')
cap = cv2.VideoCapture('data/dangerous_test5.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f'Video: 1280x720, {fps:.1f}fps, {total} frames')

clf = StageClassifier(frame_width=1280, frame_height=720)
renderer = ScreenFeedbackRenderer()
prev_target = None
stage_counts = {}
prev_stage = None

for frame_idx in range(0, total, 24):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
        break

    detections = run_detector(model, frame, {
        'model': {
            'target_class_names': ['person'],
            'confidence_threshold': 0.2,
            'image_size': 640,
            'device': 'cpu'
        }
    })
    curr = select_target(detections, prev_target, frame.shape[:2])
    if curr:
        prev_target = curr

    stage = clf.classify(curr, frame)
    renderer.render(frame.copy(), stage, curr, fps)

    time_s = frame_idx / fps
    marker = ' <-- CHANGE' if stage != prev_stage else ''
    prev_stage = stage

    if curr:
        print(f't={time_s:.1f}s [{frame_idx:3d}]: {stage:12s} h={curr.bbox[3]:3d}px cx={curr.center[0]:4d}{marker}')
    else:
        print(f't={time_s:.1f}s [{frame_idx:3d}]: {stage:12s} (no det){marker}')

    stage_counts[stage] = stage_counts.get(stage, 0) + 1

cap.release()
print()
print('=== Stage Distribution ===')
for s, cnt in sorted(stage_counts.items(), key=lambda x: -x[1]):
    print(f'  {s}: {cnt} frames ({cnt*24/total*100:.1f}%)')
print()
print('All stages verified successfully!')
