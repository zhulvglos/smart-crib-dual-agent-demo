# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 在本仓库中工作时提供指引。

## 项目概述

AI 婴儿床安全监控系统 —— 技术演示项目，验证完整闭环流程：视频输入 → YOLO 检测 → 婴儿床区域分类（安全/警告/危险动作） → 多帧防抖确认 → LangGraph 智能体决策 → 模拟语音预警 + 家长告警 → JSONL 事件日志。大部分组件（TTS、推送通知、真实摄像头）为模拟实现。

## 环境搭建

```bash
conda activate baby_system_demo
pip install -r requirements.txt
pip install ultralytics   # 项目依赖但未写入 requirements.txt
```

## 常用命令

```bash
# 依赖检查
python main.py

# 演示脚本（复杂度递增）
python demo_danger_action.py               # 仅 LangGraph 闭环（无视频）
python demo_danger_action_video.py         # 基于视频进度的规则触发
python demo_danger_action_cv.py            # OpenCV 模板匹配追踪
python demo_danger_action_detector.py      # YOLO 人体检测 + 婴儿床边界
python demo_full_stage.py                  # 完整 11 阶段分类器 + 动画叠加层

# Web 演示
cd web_demo && python start_server.py      # 服务地址 http://localhost:8080
python generate_web_demo_data.py --video data/dangerous_test1.mp4

# 验证脚本（无 pytest，均为手动验证）
python test_stage_classifier.py
python verify_demo_frames.py
python verify_danger_timeline.py
```

## 架构设计

四层递进式架构：

**第 1 层 — AI 决策（LangGraph）**：`demo_danger_action.py` 定义核心图：`detect_state → risk_decision → voice_agent → notification_agent → memory_record`。所有演示脚本均从该模块导入 `build_danger_action_graph()`、`create_mock_danger_event()` 和 `build_summary()`。

**第 2 层 — 感知检测**：`crib_detector.py` 通过 YOLO 分割模型（`yolo26n-seg.pt`）自动检测婴儿床边界，支持人体估算和硬编码矩形兜底。检测结果通过 `save_crib_config()`/`load_crib_config()` 缓存。三个 YOLO 模型文件位于项目根目录：`yolo11n.pt`、`yolo11n-pose.pt`、`yolo26n-seg.pt`。

**第 3 层 — 阶段分类**：每个演示脚本有各自的分类函数，复杂度逐级递增。`demo_full_stage.py` 拥有最完整的 `StageClassifier`，支持 11 种状态，利用姿态关键点和帧差分计算运动强度。

**第 4 层 — 可视化**：`demo_full_stage.py` 中的 `ScreenFeedbackRenderer` 为每种状态绘制动画叠加层。Web 演示（`web_demo/`）通过 HTML5 video + canvas 叠加层消费预计算的 JSON 数据。

### 关键导入链

- 所有演示脚本 → `demo_danger_action.py`（LangGraph 智能体）
- `demo_danger_action_detector.py` → `crib_detector.py`（婴儿床边界）
- `demo_full_stage.py` → `utils_chinese_text.py`（通过 PIL 在 OpenCV 帧上渲染中文）
- `generate_web_demo_data.py` → `demo_danger_action_detector.py`（检测 + 分类，用于 Web 数据生成）

## 配置文件

- `config/demo_detector_config.json` — YOLO 检测器设置：模型路径、置信度/IoU 阈值、婴儿床区域多边形角点（比例坐标）、安全/警告区域比例、防抖帧数、自动检测开关。
- `config/demo_cv_config.json` — OpenCV 模板匹配设置：ROI 搜索半径、模板更新速率、区域边界。

两个配置文件均使用比例坐标（0.0–1.0），相对于帧尺寸定义区域。

## 数据持久化

- 婴儿床边界缓存：`config/crib_config_cache.json`（自动生成）
- 危险事件日志：`logs/danger_action_events.jsonl`
- Web 演示预计算数据：`web_demo/data/`
- 验证脚本截图输出：项目根目录或 `screenshots/`
