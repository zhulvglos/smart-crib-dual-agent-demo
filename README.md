# Smart Crib Three-Agent Showcase

## 版本说明

### 新增 Growth Memory Agent 双 Agent 完整演示v2.0.0

- 完成 `危险动作 + 长记忆双 Agent Demo` 与 `语音陪伴 / 父母音色 Demo` 的双 Tab Web Showcase。
- 更新 Growth Memory Agent 的演示数据、风险趋势、记忆卡片和家长建议展示。
- 新增 Voice Companion Agent 演示链路，支持模拟哭闹事件、安抚偏好查询、浏览器 TTS 播放和 JSONL 结果回写。
- 补充 Web 端音频素材清单与本地服务接口，用于展示危险提醒、记忆卡片、建议和安抚音频。

面向智能婴儿床场景的可运行 PoC，包含三条能力链路：

```text
Dangerous Action Agent｜危险动作 Agent
→ JSONL 危险事件日志
→ Growth Memory Agent｜成长记忆 Agent
→ 风险趋势 / 记忆卡片 / 家长建议

模拟哭闹事件
→ Voice Companion Agent｜语音陪伴 Agent
↔ Growth Memory 中的安抚偏好
→ 模拟角色音色 / 浏览器 TTS / JSONL 安抚记录
```

Dangerous Action Agent 是安全看护类 Agent 的一个已验证子场景，当前项目不宣称已经覆盖趴睡、离床、坠落等完整安全场景。

Web Showcase 采用两个顶部 Tab：

- 默认页：`危险动作 + 长记忆双 Agent Demo`
- 独立页：`语音陪伴 / 父母音色 Demo`

两个 Demo 分开展示，不表达为“危险动作触发语音安抚”的串联关系。

## 当前实现

### Dangerous Action Agent｜危险动作 Agent

- 使用 YOLO11n-pose 提取人体检测框与 17 个姿态关键点。
- 结合婴儿床区域、姿态风险评分和多帧确认判断床沿风险。
- 页面显示安全、关注、危险状态及姿态判断依据。
- 危险帧经过事件去重后写入 JSONL 日志。

### Growth Memory Agent｜成长记忆 Agent

- 读取危险事件 JSONL 日志。
- 聚合每日频率、高发时段、位置分布和触发来源。
- 使用确定性规则模板生成观察摘要与家长建议。
- 当前输出属于演示数据分析，不是医疗建议或专业安全评级。

### Voice Companion Agent｜语音陪伴 Agent

- 模拟哭闹事件输入。
- 查询长期记忆中的场景安抚偏好。
- 选择妈妈音色、爸爸音色或系统默认音色策略。
- 使用规则模板生成安抚话术。
- 使用浏览器 Web Speech API 模拟播放，并可选本地生成低音量白噪音。
- 将模拟安抚结果写入 `logs/voice_companion_events.jsonl`。

当前音色为策略与交互模拟。真实产品可将 `selected_voice` 映射到外部语音克隆服务返回的 `voice_id`。

## 当前边界

- YOLO11n-pose 是通用人体姿态模型，尚未使用婴儿专用数据微调。
- 尚未接入真实硬件摄像头和大规模真实场景验证。
- Voice Companion 未接入真实 ASR、Pipecat、父母声纹训练或语音克隆服务。
- 浏览器 TTS 的实际可用音色取决于本机操作系统和浏览器。
- 推送通知、数据库、Redis、向量库均未接入。

## 运行方式

建议使用项目环境：

```bash
conda activate baby_system_demo
```

生成并验证 Voice Companion Agent：

```bash
python demo_voice_companion_agent.py
python generate_voice_companion_web_data.py
```

生成 Growth Memory Web 数据：

```bash
python generate_growth_memory_web_data.py
```

启动网页：

```bash
cd web_demo
python start_server.py
```

访问：

```text
http://localhost:8080/index.html
```

网页中的“模拟安抚完成并写入记录”需要通过 `web_demo/start_server.py` 启动，才能调用本地 POST 接口写入 JSONL。

## Voice Companion 文件

```text
demo_voice_companion_agent.py
generate_voice_companion_web_data.py
data/sample_voice/
├── crying_events.jsonl
├── voice_preferences.json
└── voice_companion_output.json
web_demo/data/voice_companion.json
logs/voice_companion_events.jsonl
```

`logs/` 为运行时数据并由 `.gitignore` 排除；`data/sample_voice/` 中的文件是可提交演示样例。

## 主要目录

```text
baby_system_demo/
├── config/
├── data/
│   ├── sample_events/
│   ├── sample_memory/
│   └── sample_voice/
├── docs/
├── logs/
├── web_demo/
├── demo_danger_action.py
├── demo_danger_action_detector.py
├── demo_growth_memory_agent.py
└── demo_voice_companion_agent.py
```
