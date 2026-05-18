"""
demo_danger_action.py

危险动作完整AI闭环Demo - 第一版

目标：
1. 模拟“宝宝靠近床沿探身”的危险动作输入
2. 通过 LangGraph 执行状态编排
3. 生成语音警示指令
4. 生成App/微信紧急告警指令
5. 记录事件日志
6. 不依赖真实摄像头、不依赖真实TTS、不依赖真实推送接口

运行方式：
python demo_danger_action.py
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import TypedDict, Dict, Any, List

from langgraph.graph import StateGraph, END


# =========================
# 1. Demo基础配置
# =========================

LOG_DIR = Path("logs")
EVENT_LOG_FILE = LOG_DIR / "danger_action_events.jsonl"


class DangerActionState(TypedDict, total=False):
    """危险动作闭环状态对象"""

    event_id: str
    state_id: str
    category: str
    source: str
    description: str
    confidence: float
    risk_level: str
    baby_position: str
    detected_at: str

    decision: Dict[str, Any]
    voice_command: Dict[str, Any]
    notification_command: Dict[str, Any]
    event_record: Dict[str, Any]

    logs: List[str]


def now_text() -> str:
    """返回当前时间字符串"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log_step(state: DangerActionState, message: str) -> DangerActionState:
    """追加流程日志"""
    logs = state.get("logs", [])
    text = f"[{now_text()}] {message}"
    logs.append(text)
    print(text)
    state["logs"] = logs
    return state


# =========================
# 2. 模拟感知输入
# =========================

def create_mock_danger_event() -> DangerActionState:
    """
    模拟视觉/传感器检测结果。

    真实产品中，这一步来自：
    - 摄像头
    - 毫米波雷达
    - 姿态识别模型
    - 多模态融合判断
    """

    return {
        "event_id": f"danger_{int(time.time())}",
        "state_id": "dangerous_action",
        "category": "危险动作",
        "source": "mock_vision_sensor",
        "description": "检测到宝宝靠近床沿并出现探身动作，存在跌落风险",
        "confidence": 0.93,
        "risk_level": "high",
        "baby_position": "near_crib_edge",
        "detected_at": now_text(),
        "logs": [],
    }


# =========================
# 3. LangGraph节点：状态识别
# =========================

def detect_state_node(state: DangerActionState) -> DangerActionState:
    """
    状态识别节点：
    判断输入是否属于危险动作。
    """

    log_step(state, "【感知输入】收到多模态检测事件")

    if state.get("state_id") == "dangerous_action":
        log_step(
            state,
            f"【状态识别】识别为危险动作 dangerous_action，置信度={state.get('confidence')}",
        )
    else:
        log_step(state, "【状态识别】未识别为危险动作")

    return state


# =========================
# 4. LangGraph节点：风险决策
# =========================

def risk_decision_node(state: DangerActionState) -> DangerActionState:
    """
    风险决策节点：
    根据状态、置信度、位置、风险等级，生成处理策略。
    """

    confidence = state.get("confidence", 0)
    baby_position = state.get("baby_position", "")
    risk_level = state.get("risk_level", "unknown")

    decision = {
        "should_intervene": False,
        "priority": "normal",
        "actions": [],
        "reason": "",
    }

    if (
        state.get("state_id") == "dangerous_action"
        and confidence >= 0.8
        and baby_position == "near_crib_edge"
    ):
        decision = {
            "should_intervene": True,
            "priority": "emergency",
            "actions": [
                "play_voice_warning",
                "send_parent_alert",
                "record_event",
            ],
            "reason": "宝宝靠近床沿探身，存在跌落风险，需要立即干预",
        }

    state["decision"] = decision

    log_step(
        state,
        f"【Agent决策】风险等级={risk_level}，优先级={decision['priority']}，是否干预={decision['should_intervene']}",
    )

    return state


# =========================
# 5. LangGraph节点：语音警示
# =========================

def voice_agent_node(state: DangerActionState) -> DangerActionState:
    """
    语音Agent节点：
    生成危险动作场景下的TTS指令。
    """

    decision = state.get("decision", {})

    if not decision.get("should_intervene"):
        log_step(state, "【语音Agent】无需语音干预")
        return state

    voice_command = {
        "type": "tts_warning",
        "state_id": "dangerous_action",
        "text": "宝贝小心，往中间来。",
        "volume": 0.85,
        "speed": 1.0,
        "repeat": 2,
        "emotion": "firm_but_gentle",
    }

    state["voice_command"] = voice_command

    log_step(
        state,
        f"【语音Agent】生成TTS警示：{voice_command['text']} 音量={voice_command['volume']} 重复={voice_command['repeat']}次",
    )

    # Demo中不真实播放，只模拟输出
    print("\n>>> TTS模拟播报")
    print(f"    {voice_command['text']}")
    print()

    return state


# =========================
# 6. LangGraph节点：通知告警
# =========================

def notification_agent_node(state: DangerActionState) -> DangerActionState:
    """
    通知Agent节点：
    生成父母端App/微信紧急告警。
    """

    decision = state.get("decision", {})

    if not decision.get("should_intervene"):
        log_step(state, "【通知Agent】无需推送通知")
        return state

    notification_command = {
        "type": "emergency_alert",
        "channels": ["app_push", "wechat"],
        "state_id": "dangerous_action",
        "title": "高风险！宝宝靠近床沿",
        "message": "检测到宝宝靠近床沿并出现探身动作，请立即查看并干预。",
        "risk_level": "high",
        "need_parent_action": True,
        "fallback": "若30秒未读，可升级为短信/电话通知",
    }

    state["notification_command"] = notification_command

    log_step(
        state,
        f"【通知Agent】生成紧急告警：{notification_command['title']}",
    )

    # Demo中不真实推送，只模拟输出
    print("\n>>> App/微信推送模拟")
    print(f"    标题：{notification_command['title']}")
    print(f"    内容：{notification_command['message']}")
    print(f"    通道：{', '.join(notification_command['channels'])}")
    print()

    return state


# =========================
# 7. LangGraph节点：事件记录
# =========================

def memory_record_node(state: DangerActionState) -> DangerActionState:
    """
    事件记录节点：
    将本次危险动作事件保存为JSONL日志，模拟成长记忆/安全事件记录。
    """

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    event_record = {
        "event_id": state.get("event_id"),
        "state_id": state.get("state_id"),
        "category": state.get("category"),
        "source": state.get("source"),
        "description": state.get("description"),
        "confidence": state.get("confidence"),
        "risk_level": state.get("risk_level"),
        "baby_position": state.get("baby_position"),
        "detected_at": state.get("detected_at"),
        "decision": state.get("decision"),
        "voice_command": state.get("voice_command"),
        "notification_command": state.get("notification_command"),
        "closed_loop_finished_at": now_text(),
    }

    state["event_record"] = event_record

    with EVENT_LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event_record, ensure_ascii=False) + "\n")

    log_step(
        state,
        f"【事件记录】已写入安全事件日志：{EVENT_LOG_FILE}",
    )

    return state


# =========================
# 8. LangGraph流程搭建
# =========================

def build_danger_action_graph():
    """
    构建危险动作闭环流程图：

    detect_state
        ↓
    risk_decision
        ↓
    voice_agent
        ↓
    notification_agent
        ↓
    memory_record
        ↓
    END
    """

    workflow = StateGraph(DangerActionState)

    workflow.add_node("detect_state", detect_state_node)
    workflow.add_node("risk_decision", risk_decision_node)
    workflow.add_node("voice_agent", voice_agent_node)
    workflow.add_node("notification_agent", notification_agent_node)
    workflow.add_node("memory_record", memory_record_node)

    workflow.set_entry_point("detect_state")

    workflow.add_edge("detect_state", "risk_decision")
    workflow.add_edge("risk_decision", "voice_agent")
    workflow.add_edge("voice_agent", "notification_agent")
    workflow.add_edge("notification_agent", "memory_record")
    workflow.add_edge("memory_record", END)

    return workflow.compile()


# =========================
# 9. 主程序入口
# =========================

def main():
    print("=" * 70)
    print("危险动作完整AI闭环Demo")
    print("场景：宝宝靠近床沿探身")
    print("=" * 70)
    print()

    # 1. 创建模拟事件
    event = create_mock_danger_event()

    print(">>> 模拟输入事件")
    print(json.dumps(event, ensure_ascii=False, indent=2))
    print()

    # 2. 构建并运行LangGraph
    app = build_danger_action_graph()

    print(">>> 开始执行AI闭环")
    print()

    final_state = app.invoke(event)

    print()
    print("=" * 70)
    print("闭环执行完成")
    print("=" * 70)

    print("\n>>> 最终结果摘要")
    summary = {
        "event_id": final_state.get("event_id"),
        "state_id": final_state.get("state_id"),
        "risk_level": final_state.get("risk_level"),
        "decision_priority": final_state.get("decision", {}).get("priority"),
        "voice_text": final_state.get("voice_command", {}).get("text"),
        "notification_title": final_state.get("notification_command", {}).get("title"),
        "event_log_file": str(EVENT_LOG_FILE),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()