# 危险动作完整AI闭环Demo

## 当前版本

PoC Rule-based Video Trigger Demo

Current trigger mode: PoC rule-based video progress trigger.

## 运行命令

```bash
python demo_danger_action_video.py
```

如果当前终端还没有进入项目环境：

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

## 说明

本 README 对应 `demo_danger_action_video.py`。该脚本用于展示危险动作 AI 闭环链路，不声称已经实现真实视觉识别。
