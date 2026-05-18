"""
Baby System Demo - 主入口

当前阶段用途：
1. 验证依赖是否安装成功
2. 验证项目目录是否可运行
3. 为后续 dangerous_action Demo 做入口准备

真正的危险动作闭环演示放在：
demo_danger_action.py
"""

import sys
from datetime import datetime


def check_dependencies():
    """检查第一阶段Demo所需依赖是否可导入"""

    dependencies = {
        "customtkinter": "GUI界面框架",
        "redis": "Redis事件总线",
        "numpy": "数值计算",
        "cv2": "OpenCV图像处理",
        "PIL": "图像处理",
        "requests": "HTTP请求",
        "aiohttp": "异步HTTP请求",
        "langgraph": "Agent状态编排",
        "langchain": "LLM应用基础框架",
        "dateutil": "日期时间解析",
    }

    print("正在检查依赖...\n")

    failed = []

    for package_name, description in dependencies.items():
        try:
            __import__(package_name)
            print(f"[OK] {package_name:<15} - {description}")
        except ImportError as e:
            print(f"[FAIL] {package_name:<15} - {description} - {e}")
            failed.append(package_name)

    print("\n依赖检查完成。")

    if failed:
        print("\n以下依赖导入失败：")
        for item in failed:
            print(f"- {item}")
        return False

    return True


def main():
    print("=" * 60)
    print("Baby System Demo")
    print("危险动作完整AI闭环Demo - 项目入口")
    print("=" * 60)
    print(f"Python版本: {sys.version}")
    print(f"启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print()

    ok = check_dependencies()

    print()
    if ok:
        print("[SUCCESS] 当前Demo基础环境可用。")
        print()
        print("下一步建议：")
        print("1. 新建 demo_danger_action.py")
        print("2. 模拟 dangerous_action 事件")
        print("3. 跑通：状态输入 → Agent决策 → 语音警示 → 通知告警 → 日志输出")
    else:
        print("[ERROR] 依赖不完整，请先修复依赖问题。")


if __name__ == "__main__":
    main()