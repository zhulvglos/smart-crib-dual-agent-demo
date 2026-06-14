# AI婴儿床监护系统 - 网页端Demo

基于YOLO目标检测的危险动作预警可视化展示

## 功能特性

- 🎥 **视频播放**：支持播放演示视频，带自定义控制栏
- 🎯 **实时检测可视化**：Canvas叠加显示YOLO检测框、安全区、警告区、危险边界
- 📊 **状态监控**：实时显示SAFE/WARNING/DANGEROUS_ACTION状态
- 📋 **事件日志**：记录危险动作触发事件
- 🎨 **科技感UI**：深色主题，适合面试展示

## 快速开始

### 1. 启动Demo服务器

```bash
cd web_demo
python start_server.py
```

服务器会自动在浏览器中打开 `http://localhost:8080`

### 2. 使用预计算数据（可选）

如果想要展示真实的YOLO检测结果，需要先运行预计算脚本：

```bash
# 在项目根目录运行
python generate_web_demo_data.py --video data/dangerous_test1.mp4
python generate_web_demo_data.py --video data/dangerous_test2.mp4
```

这会在 `web_demo/data/` 目录下生成检测数据文件。

## 文件结构

```
web_demo/
├── index.html          # 主页面
├── css/
│   └── style.css       # 样式文件
├── js/
│   └── app.js          # 主程序
├── data/               # 检测数据（预计算生成）
│   ├── dangerous_test1_detection.json
│   └── dangerous_test2_detection.json
├── start_server.py     # 启动脚本
└── README.md           # 本文件
```

## 演示模式

如果没有预计算数据，Demo会自动进入**模拟模式**，根据时间周期模拟检测状态变化：

- 0-50%：SAFE（安全区，绿色）
- 50-70%：WARNING（警告区，黄色）
- 70-100%：DANGEROUS_ACTION（危险区，红色）

## 技术架构

```
视频输入 → YOLO目标检测 → 危险边界判断 → LangGraph决策 → 语音/推送告警
```

- **检测模型**：YOLO11n预训练模型
- **危险判断**：基于边界规则的区域检测
- **防抖机制**：多帧确认策略

## 面试展示建议

1. **开场**：介绍项目背景（婴儿床监护场景）
2. **技术点**：强调YOLO检测 + 规则判断 + AI决策链
3. **演示**：播放视频，展示检测框跟随、状态变化、危险告警
4. **扩展**：提到可以接入真实摄像头、姿态识别、时序分析等

## 注意事项

- 本Demo用于技术展示，检测逻辑基于预训练YOLO模型+规则判断
- 实际产品需结合姿态识别、时序分析等技术进一步提升准确性
- 视频文件需要放在 `data/` 目录下才能正常播放
