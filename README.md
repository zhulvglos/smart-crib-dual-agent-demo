# Baby Agent Showcase

## 项目目标

危险动作 Safety Agent + 成长记忆 Growth Memory Agent 双 Agent Demo。

## Demo 核心链路

```
视频输入
→ 危险动作识别（YOLO 检测 + 区域分类 + 多帧防抖）
→ 风险判断（LangGraph Safety Agent 决策）
→ 语音提醒 / 家长告警 / 事件日志（JSONL）
→ 长记忆分析（Growth Memory Agent 读取事件日志）
→ 成长记忆卡片 / 风险趋势 / 家长建议
→ Web Showcase 展示
```

## 当前已完成

- 危险动作检测 PoC（YOLO 人体检测 + 婴儿床边界识别）
- LangGraph Safety Agent 闭环（检测 → 决策 → 语音 → 告警 → 日志）
- Web 可视化演示（视频播放 + canvas 检测叠加层 + 状态监控）
- JSONL 事件日志输出（58 条示例事件）
- 11 阶段分类器（深睡、浅醒、哭闹 L1–L3、危险动作 2–5、开心玩耍）

## 即将开发

- Growth Memory Agent（读取事件日志 → 趋势分析 → 记忆卡片生成）
- 风险趋势分析（历史事件聚合、危险模式识别）
- 家长建议生成（基于风险趋势的个性化建议）
- Web Showcase 记忆面板（成长卡片、风险趋势图、建议面板）

## 当前边界

- TTS 为模拟（生成命令参数，未实际播放语音）
- 家长告警为模拟（生成推送参数，未发送真实通知）
- 使用通用人体检测模型（YOLO11n），非婴儿专用模型
- 暂未接入真实硬件摄像头
- 暂未接入真实数据库（事件以 JSONL 文件形式存储）

## 如何运行

```bash
# 环境准备
conda activate baby_system_demo
pip install -r requirements.txt
pip install ultralytics

# 依赖检查
python main.py

# 危险动作检测演示
python demo_danger_action_detector.py

# 完整 11 阶段分类演示
python demo_full_stage.py

# Web 演示
cd web_demo && python start_server.py
# 访问 http://localhost:8080
```

## 项目结构

```
baby_system_demo/
├── src/                           # 核心代码（待整理）
│   └── safety_agent/              # 危险动作检测 Agent
├── web_demo/                      # Web 演示
├── config/                        # 检测器配置
├── data/                          # 示例数据
│   └── sample_events/             # 示例事件日志
├── docs/                          # 文档
│   ├── architecture/              # 架构设计
│   ├── demos/                     # 演示说明
│   ├── analysis/                  # 分析报告
│   └── product/                   # 产品设计
├── assets/                        # 展示素材
│   └── images/                    # 截图、预览图
├── archive/                       # 归档文件
└── logs/                          # 运行时日志
```

## 文档导航

- [项目状态总览](docs/architecture/project_status.md)
- [改进路线图](docs/architecture/roadmap.md)
- [技术约定](docs/architecture/conventions.md)
- [危险时间线分析](docs/analysis/danger_timeline_overview.md)
- [API 设计参考](docs/architecture/api_design_reference.json)
