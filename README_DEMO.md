# 危险动作完整AI闭环Demo

## 当前版本

PoC Rule-based Video Trigger Demo

Current trigger mode: PoC rule-based video progress trigger.

## 运行命令

```bash
python demo_danger_action_video.py
```

如果当前终端还没有进入项目环境，可以先运行：

```bash
conda activate baby_system_demo
python demo_danger_action_video.py
```

## Demo链路

Video Input -> Rule-based Stage Detection -> dangerous_action -> LangGraph Decision -> Voice Warning -> Parent Alert -> Event Log

## 当前能力

- 视频播放
- 安全区/危险边界叠加
- 危险动作状态触发
- AI闭环执行
- 事件日志记录

## 当前限制

- 当前不是实时视觉模型识别
- 当前目标点为模拟追踪点
- 当前TTS和App推送为模拟输出
- 当前视频触发层使用规则模拟，后端AI闭环真实执行

## 后续升级

- 目标检测
- 姿态识别
- 实时摄像头
- 多模态融合
- 真实TTS和真实App推送

## CV-assisted PoC

运行默认视频：

```bash
python demo_danger_action_cv.py
```

换新视频时，请重新选择当前视频里的目标区域：

```bash
python demo_danger_action_cv.py --video data/your_video.mp4 --select-roi
```

如果只是想更快预览：

```bash
python demo_danger_action_cv.py --speed 2.0
```

注意：这个版本不是通用婴儿姿态识别，也不是产品级视觉检测。它使用 OpenCV 目标追踪和规则危险边界，验证的是 dangerous_action 事件之后的 AI 闭环链路。

## YOLO Detection Demo

运行 YOLO 目标检测版：

```bash
python demo_danger_action_detector.py
```

切换视频：

```bash
python demo_danger_action_detector.py --video data/your_video.mp4
```

加速预览：

```bash
python demo_danger_action_detector.py --speed 2.0
```

这个版本不需要手动 ROI，会用预训练 YOLO person detector 自动生成目标框。它比 OpenCV ROI tracking 更适合换视频演示，但仍然不是产品级婴儿危险动作识别；危险判断来自配置里的边界规则。
