# 项目清理审计报告

> 生成时间：2026-06-14
> 项目路径：`D:\pythonProject1\baby_system_demo`
> 目标：为「危险动作 + 长记忆双 Agent Demo」和 Git 仓库初始化做准备

---

# 1. 当前项目总体判断

| 问题 | 判断 | 说明 |
|------|------|------|
| 当前项目主要内容是什么 | AI 婴儿床安全监控 PoC | 完整闭环：视频 → YOLO 检测 → 区域分类 → LangGraph 决策 → 模拟语音/告警 → JSONL 日志 |
| 哪些部分和危险动作 Demo 直接相关 | `demo_danger_action.py`、`demo_danger_action_detector.py`、`demo_full_stage.py`、`crib_detector.py`、`config/`、`web_demo/` | 构成完整的检测 + 决策 + 可视化链路 |
| 哪些部分和长记忆 Demo 后续相关 | `logs/danger_action_events.jsonl`、`demo_danger_action.py` 中的 `memory_record` 节点 | 当前仅有 JSONL 追加写入，无语义检索，需新建 Growth Memory Agent |
| 哪些部分是历史测试/冗余文件 | `test_stage_classifier.py`、`verify_demo_frames.py`、`verify_danger_timeline.py`、`generate_stage_previews.py`、`demo_danger_action_detector.py.backup`、`output/` | 开发过程中用于验证，不直接参与最终展示 |
| 当前是否适合直接 Git 初始化 | **不适合直接提交** | 缺少 `.gitignore` 覆盖范围检查、README 待重写、requirements.txt 缺少 `ultralytics`、存在重复视频文件 |
| 是否存在大文件/视频/模型文件需要排除 Git | **是** | `.pt` 模型 17.9 MB、`data/*.mp4` 7.9 MB、`web_demo/data/*.mp4` 3.7 MB、`output/` 30 MB，合计约 60 MB |

---

# 2. 文件分类清单

## 2.1 必须保留：核心代码

| 文件/文件夹 | 作用 | 为什么必须保留 | 后续建议位置 |
|-------------|------|----------------|--------------|
| `demo_danger_action.py` | LangGraph Agent 核心：5 节点闭环图 + mock 事件创建 + 日志写入 | 所有演示脚本的依赖基础，后续可扩展长记忆节点 | `src/safety_agent/workflow.py` |
| `demo_danger_action_detector.py` | YOLO 人体检测 + 多边形区域分类 + 多点安全检查 + 防抖 | 危险动作检测主链路 | `src/safety_agent/detector.py` |
| `demo_full_stage.py` | 11 阶段分类器 + 动画叠加层渲染器 | 最完整的分类 + 可视化实现 | `src/safety_agent/stage_classifier.py` + `src/safety_agent/renderer.py` |
| `crib_detector.py` | YOLO 分割自动检测婴儿床边界，支持缓存 | 感知层核心，三个演示脚本依赖 | `src/safety_agent/crib_detector.py` |
| `utils_chinese_text.py` | PIL 渲染中文到 OpenCV 帧 | OpenCV 演示中文显示必需 | `src/common/utils.py` |
| `generate_web_demo_data.py` | 预计算 YOLO 检测数据生成 JSON 供 Web 消费 | Web 演示数据来源 | `scripts/generate_web_demo_data.py` |
| `main.py` | 依赖检查 + 项目入口 | 快速验证环境是否就绪 | `scripts/check_deps.py` |
| `config/demo_detector_config.json` | YOLO 检测器配置（阈值、区域、防抖参数） | 检测行为的核心配置 | `config/demo_detector_config.json` |
| `config/demo_cv_config.json` | OpenCV 模板匹配配置 | CV 演示模式配置 | `config/demo_cv_config.json` |
| `web_demo/index.html` | Web 演示首页 | 网页展示入口 | `web_showcase/index.html` |
| `web_demo/js/app.js` | Web 演示核心逻辑（canvas 叠加层、状态监控） | 可视化核心 | `web_showcase/js/app.js` |
| `web_demo/css/style.css` | Web 演示样式 | 视觉呈现 | `web_showcase/css/style.css` |
| `web_demo/start_server.py` | 本地 HTTP 服务启动脚本 | 快速启动 Web 演示 | `web_showcase/start_server.py` |
| `requirements.txt` | 依赖清单 | 环境搭建必需（需补充 ultralytics） | `requirements.txt` |

## 2.2 必须保留：说明文档 / 项目资料

| 文件/文件夹 | 作用 | 对后续有什么指导意义 | 后续建议位置 |
|-------------|------|---------------------|--------------|
| `baby_system_summary.md` | 项目全貌状态文档（架构、数据流、Agent 节点、完成状态、重启指引） | **最重要的上下文文档**，后续开发新 Agent 或重构时必读 | `docs/architecture/project_status.md` |
| `IMPROVEMENT_SUGGESTIONS.md` | 改进路线图（短期/中期/长期 + 优先级表 + 模型推荐） | 指导后续功能演进方向，特别是检测精度提升和多模态融合 | `docs/architecture/roadmap.md` |
| `README_DEMO.md` | 视频触发 PoC 的说明文档（运行方式、能力、限制） | 理解当前 Demo 边界 | `docs/demos/video_poc.md` |
| `web_demo/README.md` | Web 演示说明（功能、启动、面试展示技巧） | Web 展示的使用指南 | `web_showcase/README.md` |
| `meterial/接口管理——之后.json` | 智能婴儿床后端服务完整 API 设计（14+ 模块） | 产品化蓝图，后续若加后端 API 层可参考 | `docs/architecture/api_design_reference.json` |
| `meterial/被动式功能设计t0.xlsx` | 被动式功能设计文档 | 产品设计参考 | `docs/product/passive_feature_design.xlsx` |
| `output/demo_frames/danger_analysis/overview.md` | dangerous_test5 全帧分析摘要 | 用作录屏素材和展示说明 | `docs/analysis/danger_timeline_overview.md` |
| `output/demo_frames/danger_analysis/danger_timeline_report.json` | 结构化分析报告（阶段分布、危险时间线） | 可直接作为长记忆 Agent 的输入样例 | `data/sample_events/danger_timeline_report.json` |
| `logs/danger_action_events.jsonl` | 58 条危险事件日志 | **长记忆 Agent 的核心输入数据** | `data/sample_events/danger_action_events.jsonl` |
| `.workbuddy/memory/MEMORY.md` | 关键实现约定（中文渲染、区域定义、分类优先级） | 防止踩坑的技术约定 | `docs/architecture/conventions.md` |
| `.workbuddy/memory/2026-05-21.md` | 开发日志（分类逻辑修复、全帧扫描、中文乱码修复） | 理解历史 bug 和修复方案 | `docs/changelog/2026-05-21.md` |

## 2.3 可归档：历史测试文件

| 文件/文件夹 | 当前作用 | 是否仍可参考 | 建议操作 |
|-------------|----------|-------------|----------|
| `demo_danger_action_video.py` | 最简规则触发 PoC（按视频进度分段） | 是，可作为入门演示 | 归档 |
| `demo_danger_action_cv.py` | OpenCV 模板匹配版演示 | 是，YOLO 不可用时的降级方案 | 归档 |
| `test_stage_classifier.py` | 阶段分类器快速验证 | 是，可复用验证逻辑 | 归档 |
| `verify_demo_frames.py` | 关键帧截图验证 | 是，可复用截图逻辑 | 归档 |
| `verify_danger_timeline.py` | 全帧危险时间线分析 | 是，分析报告已保留 | 归档 |
| `generate_stage_previews.py` | 生成 10 种状态效果 PNG | 是，用于验证渲染效果 | 归档 |
| `demo_danger_action_detector.py.backup` | 旧版检测器备份 | 否，已被当前版本替代 | 待确认删除 |
| `config/demo_detector_config.json.backup` | 旧版配置备份 | 否 | 待确认删除 |
| `output/pic/1-8.png` | 截图素材（来源不明） | 待确认 | 待确认删除 |
| `output/frames/frame_*.jpg` | 视频提取帧 | 是，可作为渲染测试素材 | 归档 |
| `output/demo_frames/0*_*.png` | 关键帧截图（5 张） | 是，可作为展示素材 | 归档 |
| `output/stage_previews/*.png` | 10 种状态渲染预览 | 是，可作为 README 截图 | 归档 |
| `output/demo_frames/danger_analysis/danger_*.png` | 危险帧截图（3 张） | 是，录屏素材 | 归档 |
| `logs/cv_probe_frames/*.jpg` | CV 探针帧（5 张） | 否，仅为调试产物 | 待确认删除 |
| `meterial/接口管理——之前.json` | 旧版 API 规格 | 否，已被"之后"版替代 | 待确认删除 |

## 2.4 建议排除 Git 的文件

| 文件/文件夹 | 原因 | 建议写入 .gitignore |
|-------------|------|---------------------|
| `*.pt`（yolo11n.pt、yolo11n-pose.pt、yolo26n-seg.pt） | 模型权重文件，17.9 MB，可通过 ultralytics 自动下载 | ✅ 是 |
| `data/*.mp4` | 测试视频，7.9 MB | ✅ 是 |
| `web_demo/data/*.mp4` | Web 演示视频副本，3.7 MB | ✅ 是 |
| `output/` | 生成的截图、分析报告，30 MB | ✅ 是（保留目录结构用 .gitkeep） |
| `logs/` | 运行时生成的日志和探针帧 | ✅ 是（保留目录结构用 .gitkeep） |
| `__pycache__/` | Python 缓存 | ✅ 是 |
| `.idea/` | PyCharm 配置 | ✅ 是（当前已有） |
| `.workbuddy/` | WorkBuddy 插件本地记忆 | ✅ 是 |
| `*.backup` | 备份文件 | ✅ 是 |
| `config/crib_config_cache.json` | 运行时自动生成的缓存 | ✅ 是 |

---

# 3. 和「危险动作 + 长记忆双 Agent Demo」的关系判断

**目标 Demo 结构：**
```
危险动作视频输入
→ Safety Agent 危险动作识别与风险判断
→ 语音提醒 / 家长告警 / 事件日志
→ Growth Memory Agent 读取事件日志
→ 生成成长记忆卡片 / 风险趋势 / 家长建议
→ Web Showcase 展示
```

| 文件/模块 | 参与哪个环节 | 当前是否可用 | 是否需要改造 | 改造建议 |
|-----------|-------------|-------------|-------------|----------|
| `demo_danger_action.py` | Safety Agent 工作流 | ✅ 可用 | 需要 | 在 LangGraph 图中添加 `memory_record → growth_memory_agent` 新节点；将 mock 写入改为真实 JSONL 写入 |
| `demo_danger_action_detector.py` | 危险动作检测 | ✅ 可用 | 小改 | 提取为独立模块，移除 `__main__` 中的 OpenCV 演示逻辑，保留纯函数供 Agent 调用 |
| `demo_full_stage.py` | 阶段分类 + 可视化 | ✅ 可用 | 小改 | `StageClassifier` 提取为独立模块；`ScreenFeedbackRenderer` 可供 Web 层调用 |
| `crib_detector.py` | 婴儿床边界检测 | ✅ 可用 | 无需 | 直接复用 |
| `utils_chinese_text.py` | 中文渲染 | ✅ 可用 | 无需 | 直接复用 |
| `config/demo_detector_config.json` | 检测配置 | ✅ 可用 | 无需 | 直接复用 |
| `config/demo_cv_config.json` | CV 追踪配置 | ✅ 可用 | 无需 | 直接复用 |
| `logs/danger_action_events.jsonl` | 事件日志 → 长记忆输入 | ✅ 可用 | 需要 | Growth Memory Agent 需读取此文件并做语义分析、趋势统计、卡片生成 |
| `generate_web_demo_data.py` | Web 数据预计算 | ✅ 可用 | 需要 | 扩展输出格式，增加记忆卡片、风险趋势数据 |
| `web_demo/` | Web Showcase 展示 | ⚠️ 部分可用 | 需要 | 新增记忆 Agent 展示区（成长卡片、风险趋势图、家长建议面板）；当前仅展示 Safety Agent |
| `main.py` | 依赖检查 | ✅ 可用 | 需要 | 补充 ultralytics 依赖检查 |
| `baby_system_summary.md` | 项目文档 | ✅ 可用 | 无需 | 参考用 |
| `IMPROVEMENT_SUGGESTIONS.md` | 路线图文档 | ✅ 可用 | 无需 | 参考用 |
| `README_DEMO.md` | 旧 README | ✅ 可用 | 需重写 | 合并为新的项目 README |

**新增模块（当前项目不存在）：**

| 新模块 | 作用 | 需要新建 |
|--------|------|----------|
| `memory_agent/memory_store.py` | 事件日志读取、语义索引、历史查询 | ✅ |
| `memory_agent/memory_workflow.py` | LangGraph 长记忆工作流（分析 → 趋势 → 卡片 → 建议） | ✅ |
| `memory_agent/report_generator.py` | 成长记忆卡片 + 风险趋势报告 + 家长建议生成 | ✅ |
| `memory_agent/schemas.py` | 长记忆 Agent 状态定义 | ✅ |

---

# 4. 推荐项目目录结构

```
baby-agent-showcase/
├── README.md                              # 项目说明（新写）
├── requirements.txt                       # 依赖清单（补充 ultralytics）
├── .gitignore                             # Git 忽略规则（重写）
├── CLAUDE.md                              # Claude Code 指引（已有，保留）
│
├── src/
│   ├── safety_agent/
│   │   ├── __init__.py
│   │   ├── detector.py                    # ← demo_danger_action_detector.py（提取核心逻辑）
│   │   ├── workflow.py                    # ← demo_danger_action.py（LangGraph 闭环）
│   │   ├── stage_classifier.py            # ← demo_full_stage.py 中 StageClassifier 提取
│   │   ├── crib_detector.py               # ← crib_detector.py
│   │   └── renderer.py                    # ← demo_full_stage.py 中 ScreenFeedbackRenderer 提取
│   ├── memory_agent/                      # 🆕 长记忆 Agent（新建）
│   │   ├── __init__.py
│   │   ├── memory_store.py                # 事件日志读取 + 语义索引
│   │   ├── memory_workflow.py             # LangGraph 长记忆工作流
│   │   ├── report_generator.py            # 成长卡片 / 趋势报告 / 建议生成
│   │   └── schemas.py                     # 状态类型定义
│   ├── common/
│   │   ├── __init__.py
│   │   ├── config.py                      # 配置加载
│   │   └── utils.py                       # ← utils_chinese_text.py + 通用工具
│   └── app.py                             # 统一入口，串联 Safety + Memory Agent
│
├── web_showcase/
│   ├── index.html                         # ← web_demo/index.html（扩展）
│   ├── start_server.py                    # ← web_demo/start_server.py
│   ├── js/app.js                          # ← web_demo/js/app.js（扩展）
│   ├── css/style.css                      # ← web_demo/css/style.css（扩展）
│   └── data/                              # 预计算 JSON（运行时生成，.gitignore）
│
├── config/
│   ├── demo_detector_config.json          # ← 保留
│   └── demo_cv_config.json                # ← 保留
│
├── data/
│   ├── sample_events/                     # 示例事件日志（提交到 Git）
│   │   └── danger_action_events.jsonl     # ← logs/danger_action_events.jsonl
│   ├── sample_memory/                     # 示例记忆输出（提交到 Git）
│   └── videos/                            # 测试视频（.gitignore，附下载说明）
│
├── logs/
│   └── .gitkeep                           # 运行时日志目录
│
├── output/
│   └── .gitkeep                           # 运行时输出目录
│
├── assets/
│   ├── images/                            # README 截图、展示素材
│   │   ├── stage_previews/                # ← output/stage_previews/*.png
│   │   ├── demo_frames/                   # ← output/demo_frames/*.png
│   │   └── danger_analysis/               # ← output/demo_frames/danger_analysis/*.png
│   ├── videos/                            # 录屏（.gitignore，大文件）
│   └── diagrams/                          # 架构图（待生成）
│
├── docs/
│   ├── architecture/
│   │   ├── project_status.md              # ← baby_system_summary.md
│   │   ├── roadmap.md                     # ← IMPROVEMENT_SUGGESTIONS.md
│   │   ├── conventions.md                 # ← .workbuddy/memory/MEMORY.md
│   │   └── api_design_reference.json      # ← meterial/接口管理——之后.json
│   ├── demos/
│   │   └── video_poc.md                   # ← README_DEMO.md
│   ├── analysis/
│   │   ├── danger_timeline_overview.md    # ← output/.../overview.md
│   │   └── danger_timeline_report.json    # ← output/.../danger_timeline_report.json
│   ├── changelog/
│   │   └── 2026-05-21.md                  # ← .workbuddy/memory/2026-05-21.md
│   └── product/
│       └── passive_feature_design.xlsx    # ← meterial/被动式功能设计t0.xlsx
│
├── scripts/
│   ├── check_deps.py                      # ← main.py
│   ├── generate_web_demo_data.py          # ← generate_web_demo_data.py
│   └── run_safety_demo.py                 # ← demo_danger_action_detector.py（演示入口）
│
└── archive/
    ├── old_demos/
    │   ├── demo_danger_action_video.py    # ← 归档
    │   ├── demo_danger_action_cv.py       # ← 归档
    │   └── demo_full_stage.py             # ← 归档（完整版保留参考）
    ├── old_tests/
    │   ├── test_stage_classifier.py
    │   ├── verify_demo_frames.py
    │   ├── verify_danger_timeline.py
    │   └── generate_stage_previews.py
    └── legacy_configs/
        ├── demo_detector_config.json.backup
        └── demo_danger_action_detector.py.backup
```

| 目录 | 用途 | 应放入哪些现有文件 |
|------|------|-------------------|
| `src/safety_agent/` | Safety Agent 核心代码 | `demo_danger_action.py`、`demo_danger_action_detector.py`、`demo_full_stage.py`（拆分）、`crib_detector.py` |
| `src/memory_agent/` | 长记忆 Agent（新建） | 无现有文件，需新建 |
| `src/common/` | 公共工具 | `utils_chinese_text.py` |
| `web_showcase/` | Web 展示层 | `web_demo/` 全部 |
| `config/` | 配置文件 | `config/` 保留原位 |
| `data/sample_events/` | 提交到 Git 的示例数据 | `logs/danger_action_events.jsonl` |
| `assets/images/` | 展示截图素材 | `output/` 中的 PNG 文件 |
| `docs/` | 文档 | 所有 `.md` 文档和 `.json` API 设计文件 |
| `scripts/` | 运行脚本 | `main.py`、`generate_web_demo_data.py` |
| `archive/` | 归档旧版代码 | 过期演示脚本、备份文件 |

---

# 5. 移动方案

| 原路径 | 新路径 | 操作类型 | 理由 | 风险 |
|--------|--------|----------|------|------|
| `demo_danger_action.py` | `src/safety_agent/workflow.py` | 移动 | 核心 Agent 工作流 | 需更新所有 import 路径 |
| `demo_danger_action_detector.py` | `src/safety_agent/detector.py` | 移动 | 检测主链路 | 需提取纯函数，分离 `__main__` |
| `crib_detector.py` | `src/safety_agent/crib_detector.py` | 移动 | 边界检测 | 低风险，直接移动 |
| `utils_chinese_text.py` | `src/common/utils.py` | 移动 | 公共工具 | 低风险 |
| `demo_full_stage.py` | `src/safety_agent/stage_classifier.py` + `renderer.py` | 拆分移动 | StageClassifier 和 Renderer 职责不同 | 中风险，需拆分代码 |
| `web_demo/` | `web_showcase/` | 移动 | 更清晰的命名 | 需更新 `start_server.py` 中的路径 |
| `main.py` | `scripts/check_deps.py` | 移动 | 功能定位更清晰 | 低风险 |
| `generate_web_demo_data.py` | `scripts/generate_web_demo_data.py` | 移动 | 脚本归类 | 需更新 import 路径 |
| `baby_system_summary.md` | `docs/architecture/project_status.md` | 移动 | 文档归类 | 无风险 |
| `IMPROVEMENT_SUGGESTIONS.md` | `docs/architecture/roadmap.md` | 移动 | 文档归类 | 无风险 |
| `README_DEMO.md` | `docs/demos/video_poc.md` | 移动 | 旧 README 归档 | 无风险 |
| `web_demo/README.md` | `web_showcase/README.md` | 跟随目录移动 | 随 web_demo 一起移动 | 无风险 |
| `meterial/接口管理——之后.json` | `docs/architecture/api_design_reference.json` | 移动 | 参考文档归类 | 无风险 |
| `meterial/被动式功能设计t0.xlsx` | `docs/product/passive_feature_design.xlsx` | 移动 | 产品文档归类 | 无风险 |
| `.workbuddy/memory/MEMORY.md` | `docs/architecture/conventions.md` | 移动 | 技术约定文档化 | 无风险 |
| `.workbuddy/memory/2026-05-21.md` | `docs/changelog/2026-05-21.md` | 移动 | 开发日志归档 | 无风险 |
| `output/stage_previews/*.png` | `assets/images/stage_previews/` | 移动 | 展示素材归类 | 无风险 |
| `output/demo_frames/*.png` | `assets/images/demo_frames/` | 移动 | 展示素材归类 | 无风险 |
| `output/demo_frames/danger_analysis/*.png` | `assets/images/danger_analysis/` | 移动 | 展示素材归类 | 无风险 |
| `output/frames/frame_*.jpg` | `assets/images/frames/` | 移动 | 测试素材 | 无风险 |
| `output/demo_frames/danger_analysis/danger_timeline_report.json` | `data/sample_events/danger_timeline_report.json` | 移动 | 样例数据 | 无风险 |
| `output/demo_frames/danger_analysis/overview.md` | `docs/analysis/danger_timeline_overview.md` | 移动 | 分析文档 | 无风险 |
| `logs/danger_action_events.jsonl` | `data/sample_events/danger_action_events.jsonl` | 移动 | 事件日志作为长记忆输入样例 | 无风险 |
| `config/` | `config/` | 保留原位 | 配置文件位置合理 | 无 |
| `requirements.txt` | `requirements.txt` | 保留原位 | 根目录标准位置 | 无 |
| `CLAUDE.md` | `CLAUDE.md` | 保留原位 | 根目录标准位置 | 无 |
| `demo_danger_action_video.py` | `archive/old_demos/` | 归档 | 早期 PoC，已被 detector 版替代 | 无风险 |
| `demo_danger_action_cv.py` | `archive/old_demos/` | 归档 | CV 版本，非主链路 | 无风险 |
| `test_stage_classifier.py` | `archive/old_tests/` | 归档 | 验证脚本，结果已保留 | 无风险 |
| `verify_demo_frames.py` | `archive/old_tests/` | 归档 | 验证脚本 | 无风险 |
| `verify_danger_timeline.py` | `archive/old_tests/` | 归档 | 验证脚本，报告已保留 | 无风险 |
| `generate_stage_previews.py` | `archive/old_tests/` | 归档 | 预览生成脚本 | 无风险 |
| `demo_danger_action_detector.py.backup` | `archive/legacy_configs/` | 归档 | 旧版备份 | 无风险 |
| `config/demo_detector_config.json.backup` | `archive/legacy_configs/` | 归档 | 旧版配置备份 | 无风险 |
| `meterial/接口管理——之前.json` | 归档或删除 | 待确认 | 已被"之后"版替代 | 需确认 |
| `output/pic/1-8.png` | 待确认 | 待确认 | 来源不明，不确定用途 | 需确认 |
| `logs/cv_probe_frames/` | 删除 | 待确认删除 | CV 调试产物 | 需确认 |

---

# 6. 删除/归档风险评估

| 文件 | 为什么可能冗余 | 删除风险 | 建议 |
|------|---------------|----------|------|
| `demo_danger_action_detector.py.backup` | 旧版备份，当前版本已稳定 | 低 | 可安全删除 |
| `config/demo_detector_config.json.backup` | 旧版配置备份 | 低 | 可安全删除 |
| `logs/cv_probe_frames/`（5 张 JPG） | CV 调试产物 | 低 | 可安全删除 |
| `meterial/接口管理——之前.json` | 已被"之后"版完全替代 | 低 | 可安全删除 |
| `output/pic/1-8.png` | 来源不明 | 中 | **需要确认**：是否为展示截图素材？ |
| `demo_danger_action_video.py` | 最简 PoC，逻辑已被 detector 版覆盖 | 低 | 建议归档（不删除） |
| `demo_danger_action_cv.py` | CV 追踪版，YOLO 版更优 | 低 | 建议归档（YOLO 不可用时的降级方案） |
| `output/frames/frame_*.jpg` | 视频提取帧 | 低 | 建议归档（可用作渲染测试） |
| `web_demo/data/*.mp4` | 与 `data/*.mp4` 重复 | 中 | **需确认**：Web 演示是否需要独立视频副本？移动后需更新 `app.js` 中的路径 |
| `logs/danger_action_events.jsonl` | 运行时生成 | 中 | 不删除，移动到 `data/sample_events/` 作为样例 |
| `__pycache__/` | Python 缓存 | 无 | 直接删除（.gitignore 已覆盖） |

**分级汇总：**
- **可安全删除（4 个）**：`.backup` 文件 ×2、`cv_probe_frames/`、`接口管理——之前.json`
- **建议归档（3 个）**：`demo_danger_action_video.py`、`demo_danger_action_cv.py`、`output/frames/`
- **暂不删除（2 个）**：`logs/danger_action_events.jsonl`（移动到 data/sample_events/）、`web_demo/data/*.mp4`（移动前需确认）
- **需要确认（2 个）**：`output/pic/1-8.png` 的用途、`web_demo/data/*.mp4` 是否为独立副本

---

# 7. 后续开发优先级

| 优先级 | 任务 | 说明 |
|--------|------|------|
| **P0** | Git 仓库初始化 | 完成文件清理、目录重构、.gitignore、README 后执行 `git init` |
| **P0** | 确认 JSONL 事件日志可作为长记忆输入 | 当前 `danger_action_events.jsonl` 包含完整事件结构（检测 → 决策 → 语音 → 通知），需确认字段是否满足 Growth Memory Agent 需求 |
| **P0** | 新建 Growth Memory Agent 最小模块 | `memory_store.py`（读取 JSONL）+ `memory_workflow.py`（LangGraph 图：读取 → 分析趋势 → 生成卡片 → 输出建议）+ `schemas.py` |
| **P1** | 扩展 Web Showcase | 在现有 Web 演示基础上新增「记忆面板」：成长卡片、风险趋势图、家长建议，展示双 Agent 协作 |
| **P1** | 串联双 Agent 主链路 | `app.py` 中实现：Safety Agent 输出事件 → Memory Agent 读取并生成报告 → Web 展示 |
| **P2** | 准备 60–90 秒录屏 | 内容：危险检测 → Agent 决策 → 日志 → 记忆卡片生成 → Web 展示全流程 |
| **P2** | 重写 README | 包含项目名、双 Agent 架构图、运行方式、展示截图、当前边界、后续计划 |
| **P2** | 整理 requirements.txt | 补充 `ultralytics`，移除未使用的 `customtkinter`、`redis` |

---

# 8. Git 准备建议

## 8.1 推荐 `.gitignore`

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.egg-info/
dist/
build/
*.egg

# 虚拟环境
.venv/
venv/
env/
conda-env/

# IDE
.idea/
.vscode/
*.swp
*.swo

# 项目插件
.workbuddy/
.claude/

# 模型权重（通过 ultralytics 自动下载）
*.pt
*.pth
*.onnx

# 视频文件（大文件，需单独管理）
data/videos/
*.mp4
*.avi
*.mov

# 运行时生成的日志和缓存
logs/*
!logs/.gitkeep
output/*
!output/.gitkeep
config/crib_config_cache.json

# 备份文件
*.backup

# 本地配置
.env
.env.local

# OS
.DS_Store
Thumbs.db

# 录屏大文件（可选，取决于是否需要提交）
assets/videos/*.mp4
```

## 8.2 推荐 README 结构

```markdown
# Baby Agent Showcase

AI 婴儿床安全监控系统 —— 双 Agent 协作演示

## 项目目标
（一句话说明：危险动作识别 + 成长记忆生成的端到端 AI 闭环）

## 核心链路图
（ASCII 或图片展示双 Agent 数据流）

## Agent 说明
### Safety Agent（危险动作检测）
- 输入：视频流
- 处理：YOLO 检测 → 区域分类 → 防抖确认 → LangGraph 决策
- 输出：语音提醒、家长告警、事件日志

### Growth Memory Agent（成长记忆生成）
- 输入：事件日志（JSONL）
- 处理：趋势分析 → 记忆卡片生成 → 家长建议
- 输出：成长报告、风险趋势、个性化建议

## 快速启动
（环境搭建 + 运行命令）

## 当前边界
（明确说明：模拟 TTS、模拟推送、通用人体检测、无数据库）

## 后续计划
（简要列出改进方向）

## 展示
（截图 / 录屏位置）
```

## 8.3 Git 初始化前检查清单

| 检查项 | 是否完成 | 说明 |
|--------|----------|------|
| `.gitignore` 覆盖所有大文件和生成物 | ❌ 待重写 | 当前 .gitignore 已有基础规则，需补充 |
| `requirements.txt` 完整 | ❌ 待补充 | 缺少 `ultralytics`，含未使用的 `customtkinter`、`redis` |
| 无密钥/敏感信息 | ✅ 已确认 | 项目中无 `.env`、API key 等 |
| 视频/模型文件已排除 | ❌ 待配置 | 需确认 .gitignore 规则生效 |
| `logs/` 目录有 `.gitkeep` | ❌ 待创建 | 确保目录结构提交但内容不提交 |
| `output/` 目录有 `.gitkeep` | ❌ 待创建 | 同上 |
| README.md 已编写 | ❌ 待重写 | 需从零编写双 Agent 版 README |
| 目录结构已重组 | ❌ 待执行 | 需按第 4 节方案重组 |
| 文档已归类到 `docs/` | ❌ 待执行 | 需按移动方案执行 |
| 示例数据已整理到 `data/sample_events/` | ❌ 待执行 | 事件日志移动 |
| 展示截图已整理到 `assets/images/` | ❌ 待执行 | PNG 文件移动 |
| 归档文件已移到 `archive/` | ❌ 待执行 | 旧演示脚本和测试文件 |
| 首次 `git add` 前确认文件列表 | ❌ 待执行 | `git status` 确认无意外大文件 |

---

# 9. 最终结论

**当前项目应先完成「目录重组 + 文件归类 + .gitignore 重写 + README 编写」，再执行 Git 初始化；同时保留完整的 Safety Agent 链路作为基础，新建 Growth Memory Agent 最小模块，将 `logs/danger_action_events.jsonl` 作为长记忆输入样例，最终在 Web Showcase 中展示双 Agent 协作全流程。**

---

> ⏸️ 报告已生成完毕，等待确认后再执行文件移动和目录重组。
