# AI 婴儿床监护系统 Web Showcase

## 版本说明

### 新增 Growth Memory Agent 双 Agent 完整演示v2.0.0

- 默认展示 `危险动作 + 长记忆双 Agent Demo`，聚合危险事件、成长记忆、趋势统计与建议卡片。
- 新增独立的 `语音陪伴 / 父母音色 Demo`，用于模拟哭闹事件后的安抚偏好查询、音色选择、TTS 播放和结果回写。
- Web 服务新增 Voice Companion 结果写入接口，并配套 `voice_companion.json` 与 `voice_audio_manifest.json` 演示数据。

网页使用顶部 Tab 分成两个独立 Demo：

- `危险动作 + 长记忆双 Agent Demo`：Dangerous Action Agent 与 Growth Memory Agent。
- `语音陪伴 / 父母音色 Demo`：Voice Companion Agent 与安抚偏好记忆。

默认进入危险动作双 Agent Demo。两个页面共享同一个本地服务，但不被描述为同一条业务触发链路。

## 启动

```bash
cd web_demo
python start_server.py
```

访问：

```text
http://localhost:8080/index.html
```

不要直接双击 `index.html`。Voice Companion 的 JSONL 回写依赖本地服务器提供的：

```text
POST /api/voice-companion/result
```

## Voice Companion 交互

1. 切换到“语音陪伴 / 父母音色”Tab。
2. 点击“模拟哭闹事件”。
3. 页面展示命中的夜间安抚偏好。
4. 自动选择妈妈音色，可手动切换爸爸音色或默认音色。
5. 点击“播放安抚语”调用浏览器 TTS。
6. 可开启本地浏览器生成的低音量白噪音。
7. 点击“模拟安抚完成并写入记录”。
8. 结果追加到 `logs/voice_companion_events.jsonl`。

## 模拟能力声明

- 哭闹事件为预置模拟事件。
- 音色角色通过浏览器 TTS 的语速、音高和可用系统音色模拟。
- 没有使用真人父母声音。
- 没有接入真实 ASR、Pipecat 或语音克隆服务。
- 安抚结果“3 分钟后情绪平复”为演示模拟结果。

## 数据生成

在项目根目录执行：

```bash
python generate_voice_companion_web_data.py
python generate_growth_memory_web_data.py
```

生成文件：

```text
web_demo/data/voice_companion.json
web_demo/data/growth_memory.json
```
