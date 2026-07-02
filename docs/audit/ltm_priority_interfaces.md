# 长记忆服务接口审计报告

> 目标：为 Dangerous Action Agent → Growth Memory Agent 链路找到真实可接入的长记忆接口。
> 审计时间：2026-06-22
> 审计范围：baby_system_app 后端 + apifox-openapi.json + User_Profile 事件客户端

---

## 1. 接口全景（找到的所有相关接口）

### 1.1 远程 LTM 服务（独立微服务，通过 httpx 调用）

**证据位置：** `baby_system_app/baby_bed_server_backup6.9/baby_bed_server/app/services/ltm_service.py`

LTM 服务是一个**独立的远程微服务**，`baby_bed_server` 通过 httpx 客户端调用。Base URL 配置在 `settings.LTM_API_URL`。

| # | 方法 | 远程路径 | 函数 | 请求体 | 功能 |
|---|------|---------|------|--------|------|
| L1 | POST | `/events` | `upload_event()` | `{"body_text": "描述文本"}` | 上传 MLLM 描述，自动识别事件类型/情绪/姿态 |
| L2 | GET | `/events` | `get_timeline()` | params: profile_id, date, type, limit, page | 获取统一时间轴事件列表 |
| L3 | GET | `/events/{event_id}` | `get_event_detail()` | - | 获取事件详情 |
| L4 | GET | `/memory` | `get_memory()` | - | 获取三层记忆库 L0/L1/L2 |
| L5 | GET | `/summaries/{summary_type}` | `get_daily_summary()` | summary_type: "day"/"hour" | 获取每日/每小时摘要 |
| L6 | POST | `/pipeline/extract` | `batch_extract()` | `{"descriptions": [...], "profile_id": 1}` | 批量上传 MLLM 描述 |
| L7 | POST | `/ask` | `ask()` | `{"question": "...", "profile_id": 1}` | 自然语言问答 |
| L8 | POST | `/events/delete` | `delete_event()` | `{"event_id": "..."}` | 删除事件 |
| L9 | POST | `/pipeline/build` | `rebuild_memory()` | - | 重建记忆库 |

### 1.2 baby_bed_server 本地接口

#### 长记忆存储/查询（voice 模块）

**证据位置：** `baby_system_app/baby_bed_server_backup6.9/baby_bed_server/app/api/v1/voice.py:162-188`
**Schema：** `baby_system_app/baby_bed_server_backup6.9/baby_bed_server/app/schemas/voice.py:116-140`

| # | 方法 | 路径 | 函数 | 请求体 | 功能 | 状态 |
|---|------|------|------|--------|------|------|
| B1 | POST | `/api/v1/voice/ltm/store` | `store_ltm()` | `{baby_id, content, tags[], source}` | 存储长记忆 | **TODO 占位** |
| B2 | POST | `/api/v1/voice/ltm/query` | `query_ltm()` | `{baby_id, query, limit}` | 查询长记忆 | **TODO 占位** |

#### 危险事件接口（status 模块）

**证据位置：** OpenAPI JSON 行 6669-6714，Schema 行 8118-8215
**路由文件：** `baby_system_app/baby_bed_server_backup6.9/baby_bed_server/app/api/v1/status.py`

| # | 方法 | 路径 | 函数 | 请求体 | 功能 |
|---|------|------|------|--------|------|
| B3 | POST | `/api/v1/status/danger` | create_danger_event | `DangerEventCreate` | 上报危险事件（硬件接口） |
| B4 | GET | `/api/v1/status/danger/list` | get_danger_events | params: baby_id, page, page_size | 获取危险事件列表 |
| B5 | GET | `/api/v1/status/history` | get_status_history | params: baby_id, start_date, end_date | 获取状态历史 |

#### 监测事件接口（sensor 模块）

**证据位置：** `baby_system_app/baby_bed_server_backup6.9/baby_bed_server/app/services/event_service.py`
**OpenAPI：** 行 2289-2429

| # | 方法 | 路径 | 函数 | 功能 |
|---|------|------|------|------|
| B6 | GET | `/api/v1/sensor/events` | `get_events()` | 查询监测事件列表（联合 baby_status_log/cry_event/danger_event） |
| B7 | GET | `/api/v1/sensor/events/{event_id}` | `get_event_detail()` | 查询事件详情 |
| B8 | POST | `/api/v1/sensor/events/confirm` | `confirm_event()` | 确认/标记事件 |

#### 成长报告（report 模块）

**证据位置：** `baby_system_app/baby_bed_server_backup6.9/baby_bed_server/app/services/report_service.py`

| # | 方法 | 路径 | 函数 | 功能 |
|---|------|------|------|------|
| B9 | POST | `/api/v1/milestone/generate-report` | `generate_weekly_report()` | 生成 AI 日报/周报/月报（含 LLM 生成） |
| B10 | GET | `/api/v1/milestone/reports` | `get_weekly_report_list()` | 获取报告列表 |
| B11 | GET | `/api/v1/milestone/reports/{report_id}` | `get_weekly_report_detail()` | 获取报告详情 |

---

## 2. 优先接口候选表

| 优先级 | 接口名称 | 方法 | 路径 | 请求字段 | 返回字段 | 是否适合危险动作事件 | 证据位置 |
|--------|---------|------|------|---------|---------|-------------------|---------|
| **P0** | LTM 事件入库 | POST | 远程 `/events` | `{body_text: str}` | `{event_id, message, detected_type}` | **适合** — 只需写自然语言描述，LTM 自动分类 | `ltm_service.py:23-31` |
| **P0** | LTM 时间轴查询 | GET | 远程 `/events` | params: profile_id, date, type, limit | `{items: LTMEventInfo[], total}` | **适合** — 支持 event_type="danger" 过滤 | `ltm_service.py:34-52` |
| **P0** | 危险事件上报 | POST | `/api/v1/status/danger` | `DangerEventCreate` (baby_id, device_id, danger_type, severity, detected_at...) | ApiResponse | **最适合** — 结构化字段完整匹配 | OpenAPI:6669, Schema:8118 |
| **P0** | 监测事件列表 | GET | `/api/v1/sensor/events` | params: baby_id, event_type_id=5, page | `{items[], total}` | **适合** — event_type_id=5 即 danger | `event_service.py:30-170` |
| **P1** | LTM 日摘要 | GET | 远程 `/summaries/{type}` | summary_type: "day"/"hour" | `LTMDailySummaryResponse` (danger_count 等) | **适合** — 直接有 danger_count 字段 | `ltm_service.py:71-76`, `ltm.py:116-123` |
| **P1** | LTM 记忆库 | GET | 远程 `/memory` | - | `{l0_events, l1_hourly, l2_daily}` | **适合** — 三层记忆含事件聚合 | `ltm_service.py:63-68`, `ltm.py:37-41` |
| **P1** | 本地 LTM 存储 | POST | `/api/v1/voice/ltm/store` | `{baby_id, content, tags[], source}` | ApiResponse | **适合但未实现** — voice_service 中是 TODO 占位 | `voice.py:176-188`, `voice_service.py:185-195` |
| **P1** | AI 成长报告 | POST | `/api/v1/milestone/generate-report` | `{baby_id, report_type, period_start, period_end}` | `{report_id, summary, ...}` | **适合** — 含 danger_event_count + LLM 生成报告 | `report_service.py:22-118` |
| **P2** | LTM 智能问答 | POST | 远程 `/ask` | `{question, profile_id}` | `{question, answer, related_events}` | 可选 — 面试演示加分项 | `ltm_service.py:90-98` |
| **P2** | 本地 LTM 查询 | POST | `/api/v1/voice/ltm/query` | `{baby_id, query, limit}` | ApiResponse | **未实现** — TODO 占位 | `voice.py:162-174` |
| 排除 | 作息管理 | POST/GET | `/api/v1/routine/*` | - | - | 不适用 — 纯作息计划 | `routine.py` |

---

## 3. 事件入库接口深度分析

### 3.1 两个可选入口

| 维度 | **方案 A：远程 LTM `/events`** | **方案 B：本地 `/api/v1/status/danger`** |
|------|------------------------------|----------------------------------------|
| 入库方式 | POST 自然语言文本 | POST 结构化 JSON |
| 分类方式 | LTM 自动识别 event_type/emotion/pose | 手动指定 danger_type/severity |
| 字段丰富度 | 低（只有 body_text） | 高（含 breath_rate, heart_rate, pose_status 等） |
| 外部依赖 | 依赖远程 LTM 服务可用 | 仅本地数据库 |
| 实现状态 | **已实现**（ltm_service.py 有完整调用代码） | **已实现**（OpenAPI 有完整 Schema） |
| 适合场景 | 快速对接，不改 Agent 输出格式 | 需要精确字段映射 |

### 3.2 最适合的接口：**POST `/api/v1/status/danger`**

**理由：**
1. `DangerEventCreate` Schema 字段与 Dangerous Action Agent 输出**高度匹配**
2. 不依赖外部 LTM 服务，demo 稳定性更高
3. 结构化入库后，可通过 `GET /api/v1/sensor/events?event_type_id=5` 统一查询

### 3.3 DangerEventCreate 必填字段

```json
{
  "baby_id": int,          // 必填
  "device_id": int,        // 必填
  "danger_type": str,      // 必填：breath_pause/near_edge/climbing/body_out/standing/prone_sleep/face_covered
  "severity": int,         // 必填：1=警告, 2=危险, 3=紧急
  "detected_at": datetime  // 必填：ISO 8601
}
```

### 3.4 是否支持自定义字段

| 字段 | 支持情况 |
|------|---------|
| `event_type` / `danger_type` | 支持，7 种枚举值 |
| `risk_level` → `severity` | 支持，映射为 1/2/3 整数 |
| `timestamp` → `detected_at` | 支持，ISO 8601 格式 |
| `baby_id` | 支持，必填 |
| `pose_status` | 支持，可选字段 |
| 自定义 trigger_mode | **不支持**，Schema 中无此字段 |

### 3.5 是否可写入不同事件类型

**是**。Schema 明确支持 7 种 danger_type：
- `breath_pause`（呼吸暂停）
- `near_edge`（靠近床边）
- `climbing`（翻床）
- `body_out`（身体探出）
- `standing`（站立）
- `prone_sleep`（趴睡）
- `face_covered`（捂口鼻）

哭闹、睡眠等走独立接口（`/api/v1/status/cry`、状态日志）。

---

## 4. 字段映射表

### Dangerous Action Agent JSONL → DangerEventCreate

| Dangerous Action Agent 字段 | 长记忆接口字段 | 转换规则 | 是否丢失信息 |
|---------------------------|--------------|---------|------------|
| `event_id` | — | 丢弃（数据库自动生成 ID） | 是（但可用 note 字段保留） |
| `state_id` | — | 丢弃 | 是（固定值，信息量低） |
| `risk_level: "high"` | `severity: 2` | 映射：low→1, medium→2, high→3 | 否（精确映射） |
| `trigger_mode: "yolo_person_detection_boundary"` | `danger_type: "near_edge"` | 规则映射（见下表） | 是（丢失触发模式细节） |
| `detected_at` | `detected_at` | 直接传递，ISO 8601 | 否 |
| `target_bbox` | `body_offset_cm` | 可选：计算偏移距离 | 信息降级 |
| `target_center` | `position_x/y` | 直接映射 x→position_x, y→position_y | 否 |
| `confidence` | `note` 字段追加 | 存入备注 | 是（非结构化） |
| `frame_index` | `note` 字段追加 | 存入备注 | 是（非结构化） |
| `voice_command` | — | 丢弃（语音指令不属于入库内容） | 是（但非核心） |
| `notification_command` | — | 丢弃 | 是（但非核心） |
| `baby_id` | `baby_id` | 必须传入固定测试 ID | 否 |
| `device_sn` | `device_id` | 需要先查设备表获取 device_id | **需适配** |

### trigger_mode → danger_type 映射规则

| trigger_mode | danger_type | 说明 |
|-------------|-------------|------|
| `yolo_person_detection_boundary` | `near_edge` | YOLO 检测到靠近边界 |
| `cv_target_tracking_boundary` | `near_edge` | OpenCV 追踪到靠近边界 |
| `rule_based_video_progress` | `near_edge` | 视频规则触发 |
| `mock_event` / `mock_vision_sensor` | `near_edge` | 模拟事件（统一映射） |

---

## 5. 最小可接入闭环（3 个接口）

| 角色 | 最终选用接口 | 为什么选它 |
|------|------------|----------|
| **危险动作事件入库** | `POST /api/v1/status/danger` | 结构化字段完整匹配，含 danger_type/severity/detected_at/pose_status，不依赖外部 LTM 服务，demo 稳定 |
| **查询事件列表** | `GET /api/v1/sensor/events?event_type_id=5` | 共用接口，联合查询 cry_event + danger_event + baby_status_log，支持分页和按类型过滤 |
| **获取日摘要/报告** | `POST /api/v1/milestone/generate-report` | 自动生成含 danger_event_count 的 AI 报告，内嵌 LLM 生成 summary/highlights/recommendations |

### 备选闭环（纯 LTM 路径）

| 角色 | 接口 | 说明 |
|------|------|------|
| 事件入库 | `POST` 远程 `/events` | 只需 body_text，自动分类 |
| 查询记忆 | `GET` 远程 `/memory` | L0/L1/L2 三层结构 |
| 日摘要 | `GET` 远程 `/summaries/day` | 含 danger_count |

---

## 6. 接入条件判断

### 方案 A：本地 `/api/v1/status/danger`

| 项目 | 结论 |
|------|------|
| 是否需要登录 Token | **需要** — OpenAPI 中 status 模块需要 Authorization header |
| 是否需要 device_sn/device_id | **需要** — `device_id` 为必填字段 |
| 是否需要 baby_id | **需要** — 必填字段 |
| 是否需要 family_id | **不需要** — Schema 中无此字段 |
| 是否可用 demo_baby_001 测试 | **需要先在数据库中创建测试宝宝和设备记录** |
| 是否适合 Python Adapter 调用 | **非常适合** — 标准 REST + JSON |
| 是否适合网页直接调用 | **需要** — 需要 Token 认证，需后端代理或直连 |

### 方案 B：远程 LTM `/events`

| 项目 | 结论 |
|------|------|
| 是否需要登录 Token | **不需要** — ltm_service.py 中无 Token 传递 |
| 是否需要 device_sn | **不需要** |
| 是否需要 baby_id | **不需要** — 用 profile_id（默认值 1） |
| 是否需要 family_id | **不需要** |
| 是否可用 demo_baby_001 测试 | **可以用 profile_id=1** |
| 是否适合 Python Adapter 调用 | **非常适合** — 一行 `upload_event(description)` |
| 是否适合网页直接调用 | **不适合** — 需要后端代理（httpx 调用） |

---

## 7. 特别说明：`/api/v1/voice/ltm/store` 和 `/api/v1/voice/ltm/query`

**这是 baby_bed_server 本地的长记忆接口，但当前是 TODO 占位。**

证据：
- `voice_service.py:185-195` — `store_ltm()` 返回硬编码 mock 数据
- `voice_service.py:170-182` — `query_ltm()` 返回硬编码 mock 数据
- 路由已注册（voice.py:162-188），OpenAPI 已定义（行 1689-1807）
- Schema 完整：`LTMStoreRequest{baby_id, content, tags[], source}` + `LTMQueryRequest{baby_id, query, limit}`

**如果同事正在开发 LTM 服务的真实实现**，这两个接口未来可以成为最合适的入口：
- `POST /api/v1/voice/ltm/store` — 结构化存储，支持 tags 和 source 字段
- `POST /api/v1/voice/ltm/query` — 语义查询

**但目前不建议依赖它们做 demo**。

---

## 8. 最终建议

> **最推荐接入 Dangerous Action Agent 的接口是：`POST /api/v1/status/danger`**
> - 结构化字段完整匹配危险动作事件
> - 已有 OpenAPI Schema 和数据模型，可直接对接
> - 需要：baby_id + device_id + danger_type + severity + detected_at
>
> **最推荐读取 Growth Memory 展示数据的接口是：`GET /api/v1/sensor/events?event_type_id=5`**
> - 共用查询接口，自动联合 danger_event 表
> - 支持分页、按宝宝/设备/类型过滤
>
> **备选方案（如果远程 LTM 服务可用）：`POST 远程 /events` + `GET 远程 /summaries/day`**
> - 只需 body_text，零字段映射
> - 自带日摘要能力
>
> **是否可以进入下一步接口联调：需要确认以下前置条件**
> 1. baby_bed_server 是否已部署并可访问
> 2. 是否有测试用的 baby_id 和 device_id（或需要先创建）
> 3. 认证 Token 如何获取（是否可绕过，或需要先调 `/api/v1/auth/login`）
> 4. 远程 LTM 服务（`settings.LTM_API_URL`）是否可访问

---

## 附录：证据文件索引

| 文件 | 路径 | 核心内容 |
|------|------|---------|
| LTM 服务客户端 | `baby_system_app/baby_bed_server_backup6.9/baby_bed_server/app/services/ltm_service.py` | 9 个远程 API 方法 |
| LTM Schema | `.../app/schemas/ltm.py` | LTMTimelineRequest/Response, LTMEventUploadRequest, LTMDailySummaryResponse |
| 危险事件模型 | `.../app/models/danger_event.py` | DangerEvent ORM（7 种 danger_type, severity 1-3） |
| 危险事件 Schema | OpenAPI JSON 行 8118-8215 | DangerEventCreate 必填字段 |
| 事件服务 | `.../app/services/event_service.py` | 联合查询 cry/danger/status_log |
| 报告服务 | `.../app/services/report_service.py` | AI 日报/周报含 danger_event_count + LLM |
| 语音模块 LTM | `.../app/api/v1/voice.py:162-188` | /ltm/store + /ltm/query（TODO 占位） |
| 语音 Schema | `.../app/schemas/voice.py:116-140` | LTMStoreRequest + LTMQueryRequest |
| OpenAPI 文档 | `baby_system_app/apifox-openapi.json/默认模块.openapi.json` | 全部接口定义 |
| User_Profile 事件客户端 | `baby_system_app/User_Profile_20260615/User_Profile/utils/event_client.py` | 硬件事件接口调用示例 |
