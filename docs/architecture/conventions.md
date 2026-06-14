# MEMORY.md — 项目长期记忆

## 项目约定

### 中文字体渲染
- OpenCV `cv2.putText()` 不支持中文，所有中文必须使用 `utils_chinese_text.cv2_puttext_cn()` 渲染。
- 该函数基于 PIL/Pillow + 系统字体（优先 msyh.ttc 微软雅黑）。
- 坐标体系使用 `anchor="lt"`（左上角），与 cv2.putText 的 baseline 不同，纵坐标需相应调整。
- 新脚本如需渲染中文，必须导入并使用此工具函数。

### 婴儿床边界
- 使用 `crib_detector.py` 中的 `load_crib_config()` / `save_crib_config()` 持久化。
- 路径比对需要 `os.path.normpath()` + `.replace("\\", "/")` 归一化。
- 三区定义：safe_contour（绿）→ warning_contour → crib_contour（红）。

### 阶段分类
- 优先级：危险动作 > 哭闹等级 > 苏醒/熟睡 > 高兴玩耍
- 使用 YOLO11-pose 模型（yolo11n-pose.pt）提取 17 点姿态关键点。
- 危险动作确认需要 danger_confirm_frames=5 帧去抖。
