# Baby System Demo 项目状态总结

本文档用于项目重启后的上下文交接，帮助 ChatGPT 快速了解当前 `baby_system_demo` 的目标、架构、代码结构和完成情况。

## 项目目标

本项目是一个“AI 婴儿床安全监护系统”PoC/demo，核心目标是演示：

1. 从婴儿床监控视频中识别人体/婴儿目标。
2. 基于安全区、警告区、危险边界判断婴儿是否靠近或越出床边。
3. 当检测到 `DANGEROUS_ACTION` 时，触发 LangGraph Agent 闭环。
4. Agent 闭环生成语音提醒、家长 App/微信告警，并写入事件日志。
5. 提供 OpenCV 桌面演示和网页端可视化演示。

当前项目重点是技术演示，不是生产级产品。视觉检测、危险规则、TTS、App 推送等仍有模拟成分。

## 系统架构

当前主链路可以理解为：

```text
视频输入
  -> YOLO / YOLO-pose 人体检测
  -> 婴儿床区域/安全区/警告区/危险边界判断
  -> SAFE / WARNING / DANGEROUS_ACTION 阶段分类
  -> 多帧防抖确认
  -> LangGraph Agent 决策闭环
  -> 模拟 TTS 语音提醒
  -> 模拟 App / 微信家长告警
  -> JSONL 事件日志
```

主要演示形态：

1. 命令行 Agent 闭环：`demo_danger_action.py`
2. 规则触发视频演示：`demo_danger_action_video.py`
3. OpenCV 模板跟踪演示：`demo_danger_action_cv.py`
4. YOLO 检测 + 危险边界演示：`demo_danger_action_detector.py`
5. 多阶段分类 + 屏幕反馈演示：`demo_full_stage.py`
6. 网页可视化演示：`web_demo/`

## 目录结构

```text
baby_system_demo/
  main.py                              # 依赖检查和项目入口提示
  demo_danger_action.py                # LangGraph 危险动作 Agent 闭环
  demo_danger_action_video.py          # 基于视频进度规则触发的闭环演示
  demo_danger_action_cv.py             # OpenCV 模板跟踪版本演示
  demo_danger_action_detector.py       # YOLO 人体检测 + 区域规则 + Agent 闭环
  demo_full_stage.py                   # 多阶段分类、YOLO-pose、屏幕反馈完整演示
  crib_detector.py                     # 婴儿床区域自动/降级检测模块
  generate_web_demo_data.py            # 为网页端预计算每帧检测数据
  generate_stage_previews.py           # 生成阶段预览图
  test_stage_classifier.py             # 阶段分类快速验证脚本
  verify_danger_timeline.py            # 危险时间线验证脚本
  verify_demo_frames.py                # demo 关键帧验证脚本
  utils_chinese_text.py                # OpenCV 中文文字绘制工具
  requirements.txt                     # Python 依赖列表

  config/
    demo_detector_config.json          # YOLO 检测版本配置
    demo_cv_config.json                # OpenCV 模板跟踪版本配置

  data/
    dangerous_test1.mp4 ...            # 测试视频素材

  logs/
    danger_action_events.jsonl         # Agent 闭环事件日志

  output/
    demo_frames/                       # 验证/导出帧
    frames/
    pic/
    stage_previews/

  web_demo/
    index.html                         # 网页 demo 页面
    css/style.css                      # 网页样式
    js/app.js                          # 网页端视频、canvas、状态、事件逻辑
    start_server.py                    # 静态 HTTP 服务，默认 8080
    data/
      dangerous_test*.mp4              # 网页端演示视频
      dangerous_test*_detection.json   # 预计算检测数据

  models/
    yolo11n.pt                         # YOLO 普通检测模型，实际位于项目根目录
    yolo11n-pose.pt                    # YOLO 姿态模型，实际位于项目根目录
    yolo26n-seg.pt                     # YOLO 分割模型，实际位于项目根目录
```

注意：部分 README、HTML 文案和注释出现中文编码错乱，但 Python 代码主逻辑仍可读、可运行。

## 数据库

当前没有正式数据库。

现有数据持久化方式：

1. 事件日志：`logs/danger_action_events.jsonl`
   - 由 `demo_danger_action.py` 的 `memory_record_node()` 追加写入。
   - 每行是一个危险事件 JSON。
   - 字段包括 `event_id`、`state_id`、`risk_level`、`trigger_mode`、`video_path`、`frame_index`、`target_bbox`、`decision`、`voice_command`、`notification_command` 等。

2. 检测配置：`config/*.json`
   - `demo_detector_config.json`：YOLO 模型路径、目标类别、置信度、安全区、警告区、危险边界、防抖帧数等。
   - `demo_cv_config.json`：OpenCV 模板跟踪版本的安全区、警告区、危险边界和跟踪参数。

3. 网页预计算数据：`web_demo/data/*_detection.json`
   - 由 `generate_web_demo_data.py` 生成。
   - 包含视频信息、几何区域、每帧阶段、目标框、置信度、事件列表。

4. Redis 只在 `requirements.txt` 和 `main.py` 依赖检查里出现，当前没有实际接入事件总线。

如果后续要产品化，建议新增 SQLite/PostgreSQL 表来存储设备、视频、检测帧、事件、告警、用户确认记录。

## 接口

当前没有 FastAPI/Flask/Django 等后端 API。

已有“接口”主要是脚本入口和静态文件服务：

1. 命令行入口

```bash
python main.py
python demo_danger_action.py
python demo_danger_action_video.py
python demo_danger_action_cv.py
python demo_danger_action_detector.py
python demo_full_stage.py
python generate_web_demo_data.py --video data/dangerous_test1.mp4
```

2. 网页 demo 静态服务

```bash
cd web_demo
python start_server.py
```

默认访问：

```text
http://localhost:8080/index.html
```

`web_demo/start_server.py` 使用 Python 标准库 `ThreadingHTTPServer` 提供静态文件访问，支持局域网访问，但没有业务 API、登录、上传、数据库查询等后端能力。

3. 前端数据读取

网页端 `web_demo/js/app.js` 会读取：

```text
web_demo/data/dangerous_test1.mp4
web_demo/data/dangerous_test1_detection.json
web_demo/data/dangerous_test2.mp4
web_demo/data/dangerous_test2_detection.json
web_demo/data/dangerous_test3.mp4
web_demo/data/dangerous_test3_detection.json
```

页面通过 `<video>` 播放视频，并用 `<canvas>` 叠加检测框、安全区、警告区、危险边界和状态信息。

## Agent 逻辑

Agent 闭环集中在 `demo_danger_action.py`。

状态对象：

```text
DangerActionState
```

核心字段：

```text
event_id
state_id
category
source
description
confidence
risk_level
baby_position
trigger_mode
video_path
frame_index
target_center
target_bbox
decision
voice_command
notification_command
event_record
logs
```

LangGraph 节点顺序：

```text
detect_state
  -> risk_decision
  -> voice_agent
  -> notification_agent
  -> memory_record
  -> END
```

节点职责：

1. `detect_state_node`
   - 接收感知事件。
   - 判断 `state_id` 是否为 `dangerous_action`。

2. `risk_decision_node`
   - 根据 `state_id`、`confidence`、`baby_position`、`risk_level` 决定是否干预。
   - 当前规则：`state_id == dangerous_action` 且 `confidence >= 0.8` 且 `baby_position == near_crib_edge` 时触发干预。
   - 触发后 actions 包括 `play_voice_warning`、`send_parent_alert`、`record_event`。

3. `voice_agent_node`
   - 生成模拟 TTS 命令。
   - 当前只打印模拟语音内容，没有接真实 TTS 设备。

4. `notification_agent_node`
   - 生成模拟 App/微信告警。
   - 当前只打印模拟推送内容，没有接真实推送服务。

5. `memory_record_node`
   - 将闭环结果写入 `logs/danger_action_events.jsonl`。

视觉检测触发 Agent 的位置主要在：

```text
demo_danger_action_detector.py
demo_danger_action_cv.py
demo_danger_action_video.py
```

这些脚本在判定 `DANGEROUS_ACTION` 后，会调用：

```python
build_danger_action_graph()
create_mock_danger_event(...)
app.invoke(event)
```

也就是说，Agent 输入虽然带有视频帧、目标框、检测置信度等字段，但事件对象仍是基于 `create_mock_danger_event()` 扩展出来的 demo 事件。

## 完成情况

已完成：

1. 基础项目结构和依赖检查入口。
2. LangGraph 危险动作闭环：
   - 状态识别
   - 风险决策
   - 模拟语音提醒
   - 模拟家长告警
   - JSONL 日志记录
3. 多种视觉演示脚本：
   - 视频进度规则触发
   - OpenCV 模板跟踪
   - YOLO person 检测
   - YOLO-pose 关键点辅助
   - 婴儿床区域/安全区/警告区/危险边界叠加
4. 多阶段分类雏形：
   - 熟睡
   - 苏醒
   - 哭闹 1/2/3 级
   - 危险动作多个等级
   - 高兴玩耍
5. 网页端 demo：
   - 视频播放
   - canvas 检测框/区域叠加
   - SAFE/WARNING/DANGEROUS_ACTION 状态显示
   - 事件日志展示
   - 预计算检测 JSON 加载
6. 测试/验证素材：
   - `data/dangerous_test1.mp4` 到 `dangerous_test5.mp4`
   - `web_demo/data/*_detection.json`
   - `logs/danger_action_events.jsonl`

未完成/仍是模拟：

1. 没有正式数据库。
2. 没有后端 API 服务。
3. 没有真实摄像头实时流接入。
4. 没有真实 TTS 播放设备接入。
5. 没有真实 App、微信、短信或电话推送接入。
6. 没有用户系统、设备系统、告警确认流程。
7. 没有生产级婴儿专用检测模型，目前主要使用通用 `person` 检测。
8. YOLO/pose 判断仍以规则为主，准确性依赖视频角度、配置区域和模型检测质量。
9. 部分中文文档和网页文案存在编码错乱，需要统一修复为 UTF-8。
10. `requirements.txt` 未显式列出 `ultralytics`，但多个脚本实际依赖它。

## 建议重启后的优先事项

1. 先确认运行环境：

```bash
conda activate baby_system_demo
python main.py
```

2. 安装/补齐视觉依赖：

```bash
pip install ultralytics
```

3. 快速跑 Agent 闭环：

```bash
python demo_danger_action.py
```

4. 快速跑 YOLO 检测演示：

```bash
python demo_danger_action_detector.py
```

5. 快速跑网页演示：

```bash
cd web_demo
python start_server.py
```

6. 后续开发建议：
   - 修复中文编码错乱。
   - 把 `ultralytics` 加入 `requirements.txt`。
   - 统一主入口，明确推荐 demo 脚本。
   - 加一个 FastAPI 后端，将事件日志、检测结果、告警状态暴露为 API。
   - 加 SQLite/PostgreSQL 存储事件和告警。
   - 将模拟 TTS/推送替换为真实服务适配器。
   - 优先使用 `yolo11n-pose.pt` 做姿态关键点判断，减少只靠 bbox 的误判。
