"""
demo_growth_memory_agent.py

Growth Memory Agent - Long-term memory analysis for smart crib safety system.

Reads JSONL event logs from the Safety Agent, performs trend analysis, and generates
parent-facing "growth memory" cards, risk trends, and actionable suggestions.
All insights use natural language suitable for parents, not technical jargon.

When MIMO_API_KEY is set, calls mimo-v2.5-pro to generate richer, non-template
insights. Falls back to built-in templates if unavailable.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, StateGraph

# Optional: DeepSeek via OpenAI-compatible API
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore[assignment,misc]

SAMPLE_EVENTS_FILE = Path("data/sample_events/danger_action_events.jsonl")
WEB_OUTPUT_FILE = Path("web_demo/data/growth_memory.json")
MEMORY_OUTPUT_FILE = Path("data/sample_memory/growth_memory_output.json")


class GrowthMemoryState(TypedDict, total=False):
    """State object passed through the Growth Memory Agent graph."""

    # Phase A: load_events
    events_file: str
    events: List[Dict[str, Any]]
    event_count: int
    date_range: Dict[str, str]

    # Phase B: analyze_trends
    daily_frequency: Dict[str, int]
    hourly_distribution: Dict[int, int]
    peak_hours: List[int]
    confidence_trend: Dict[str, Any]
    risk_distribution: Dict[str, int]
    position_heatmap: Dict[str, int]
    trigger_breakdown: Dict[str, int]

    # Phase C: generate_insights
    memory_cards: List[Dict[str, Any]]
    parent_suggestions: List[Dict[str, Any]]
    summary_stats: Dict[str, Any]
    llm_used: bool

    # Phase D: render_output
    output: Dict[str, Any]
    logs: List[str]


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log_step(state: GrowthMemoryState, message: str) -> GrowthMemoryState:
    logs = state.get("logs", [])
    text = f"[{now_text()}] {message}"
    logs.append(text)
    print(text)
    state["logs"] = logs
    return state


# ── Node A: load_events ──────────────────────────────────────────────

def get_event_timestamp(event: dict) -> str:
    """Support both danger-event and shared Agent event timestamps."""
    return event.get("detected_at") or event.get("timestamp") or ""


def load_events_node(state: GrowthMemoryState) -> GrowthMemoryState:
    """Read JSONL events, sort by time, populate metadata."""
    log_step(state, "[Growth Memory] Loading event logs...")

    events_file = state.get("events_file", str(SAMPLE_EVENTS_FILE))
    events = []

    with open(events_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))

    events.sort(key=get_event_timestamp)

    dates = [get_event_timestamp(e)[:10] for e in events if get_event_timestamp(e)]

    state["events"] = events
    state["event_count"] = len(events)
    state["date_range"] = {
        "start": dates[0] if dates else "N/A",
        "end": dates[-1] if dates else "N/A",
    }

    log_step(state, f"[Growth Memory] Loaded {len(events)} events, "
             f"date range: {state['date_range']['start']} ~ {state['date_range']['end']}")
    return state


# ── Node B: analyze_trends ───────────────────────────────────────────

def analyze_trends_node(state: GrowthMemoryState) -> GrowthMemoryState:
    """Compute statistical trends from event data."""
    log_step(state, "[Growth Memory] Analyzing trends...")

    events = state.get("events", [])
    if not events:
        log_step(state, "[Growth Memory] No events to analyze.")
        return state

    # Daily frequency
    daily: Counter = Counter()
    for e in events:
        day = get_event_timestamp(e)[:10]
        if day:
            daily[day] += 1

    # Hourly distribution
    hourly: Counter = Counter()
    for e in events:
        try:
            hour = int(get_event_timestamp(e)[11:13])
            hourly[hour] += 1
        except (ValueError, IndexError):
            pass

    peak_hours = [h for h, _ in hourly.most_common(3)]
    peak_hours.sort()

    # Confidence trend
    confidences = [e.get("confidence", 0) for e in events if e.get("confidence")]
    confidence_trend = {}
    if confidences:
        confidence_trend = {
            "average": round(sum(confidences) / len(confidences), 4),
            "min": round(min(confidences), 4),
            "max": round(max(confidences), 4),
            "trend": "rising" if len(confidences) > 1 and confidences[-1] > confidences[0] else "stable",
        }

    # Risk distribution
    risk_dist: Counter = Counter()
    for e in events:
        risk_dist[e.get("risk_level", "unknown")] += 1

    # Position heatmap
    pos_dist: Counter = Counter()
    for e in events:
        pos_dist[e.get("baby_position", "unknown")] += 1

    # Trigger mode breakdown
    trigger_dist: Counter = Counter()
    for e in events:
        mode = e.get("trigger_mode", e.get("source", "unknown"))
        friendly = _friendly_trigger_name(mode)
        trigger_dist[friendly] += 1

    state["daily_frequency"] = dict(sorted(daily.items()))
    state["hourly_distribution"] = dict(sorted(hourly.items()))
    state["peak_hours"] = peak_hours
    state["confidence_trend"] = confidence_trend
    state["risk_distribution"] = dict(risk_dist)
    state["position_heatmap"] = dict(pos_dist)
    state["trigger_breakdown"] = dict(trigger_dist)

    log_step(state, f"[Growth Memory] Trends: {len(daily)} days, "
             f"peak hours={peak_hours}, avg confidence={confidence_trend.get('average')}")
    return state


def _friendly_trigger_name(mode: str) -> str:
    """Map technical trigger modes to parent-friendly names."""
    mapping = {
        "mock_event": "模拟感知",
        "mock_vision_sensor": "模拟视觉感知",
        "rule_based_video_progress": "视频规则触发",
        "cv_target_tracking_boundary": "OpenCV 追踪",
        "yolo_person_detection_boundary": "AI 人体检测",
        "video_demo": "视频演示",
        "cv_assisted_video_demo": "CV 辅助检测",
        "yolo_person_detector_video_demo": "YOLO 智能检测",
    }
    return mapping.get(mode, mode)


# ── LLM Integration (MiMo / mimo-v2.5-pro) ─────────────────────────

LLM_MODEL = "mimo-v2.5-pro"
LLM_BASE_URL = "https://token-plan-cn.xiaomimimo.com/v1"


def _build_llm_prompt(state: GrowthMemoryState) -> str:
    """Build a Chinese prompt from trend statistics for the LLM."""
    event_count = state.get("event_count", 0)
    date_range = state.get("date_range", {})
    daily = state.get("daily_frequency", {})
    peak_hours = state.get("peak_hours", [])
    conf = state.get("confidence_trend", {})
    risk = state.get("risk_distribution", {})
    positions = state.get("position_heatmap", {})
    triggers = state.get("trigger_breakdown", {})

    # Pre-compute summary stats for context
    high_risk = risk.get("high", 0)
    num_days = len(daily) if daily else 1
    safety_score = max(0, min(100, 100 - (high_risk / max(num_days, 1)) * 3))

    stats_context = json.dumps({
        "event_count": event_count,
        "date_range": date_range,
        "days_covered": num_days,
        "safety_score": round(safety_score),
        "daily_frequency": daily,
        "peak_hours": peak_hours,
        "confidence_trend": conf,
        "risk_distribution": risk,
        "position_heatmap": positions,
        "trigger_breakdown": triggers,
    }, ensure_ascii=False, indent=2)

    prompt = f"""你是一位严谨的婴幼儿安全数据分析助手。以下是智能婴儿床 PoC 的演示事件数据，请基于这些数据生成面向家长的观察摘要。

## 统计数据
{stats_context}

## 要求
请生成 JSON 格式的分析结果，包含以下两个字段：

1. "memory_cards": 成长记忆卡片数组（2-4张），每张包含：
   - "title": 简短标题（10字以内）
   - "body": 详细描述（60-120字），用温暖、专业的语气，基于真实数据给出具体分析
   - "icon": 图标类型，可选 "exploration" | "risk" | "tech" | "positive" | "sleep" | "milestone"
   - "severity": "info" | "warning" | "positive"

2. "parent_suggestions": 家长建议数组（2-4条），每条包含：
   - "title": 建议标题（10字以内）
   - "body": 建议内容（40-80字），具体可执行
   - "priority": "high" | "medium" | "low"

注意：
- 所有内容使用中文
- 语气温和、清晰、专业，适合新手父母阅读
- 数据要引用具体的数字
- 只描述数据中直接观察到的现象，不把相关性写成因果关系
- 不得根据事件次数下降推断宝宝的安全感、适应能力、情绪或发育水平发生变化
- 不得根据检测置信度推断误报率、准确率或产品可靠性
- 明确说明结论来自短周期演示数据，仍需结合每日监控时长、更长周期和真实场景验证
- 建议仅限一般安全观察，不给出医疗、诊断或临床结论
- 请直接返回 JSON，不要有其他文字

返回格式（严格 JSON）：
{{"memory_cards": [...], "parent_suggestions": [...]}}"""

    return prompt


def _try_parse_json(text: str) -> Optional[Dict[str, Any]]:
    """Attempt to parse JSON, including truncated responses from reasoning models."""
    try:
        data = json.loads(text)
        if isinstance(data.get("memory_cards"), list) and isinstance(
            data.get("parent_suggestions"), list
        ):
            return data
    except (json.JSONDecodeError, KeyError):
        pass

    # Try repairing truncated JSON by closing open brackets
    open_braces = text.count("{") - text.count("}")
    open_brackets = text.count("[") - text.count("]")
    if open_braces > 0 or open_brackets > 0:
        # Remove trailing incomplete string/value
        repaired = text.rstrip()
        if repaired.endswith(","):
            repaired = repaired[:-1]
        # Close any incomplete string
        if repaired.count('"') % 2 != 0:
            repaired += '"'
        repaired += "]" * open_brackets + "}" * open_braces
        try:
            data = json.loads(repaired)
            if isinstance(data.get("memory_cards"), list) and isinstance(
                data.get("parent_suggestions"), list
            ):
                return data
        except (json.JSONDecodeError, KeyError):
            pass

    return None


def _parse_llm_response(text: str) -> Optional[Dict[str, Any]]:
    """Parse JSON from LLM response text.

    Handles reasoning-model output (<think>...</think> tags, markdown fences,
    and text before/after the JSON block).
    """
    text = text.strip()

    # Strip <think>...</think> reasoning tags (used by MiMo and other reasoning models)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # Strip ```json ... ``` wrapper
    if text.startswith("```"):
        first_newline = text.find("\n")
        last_fence = text.rfind("```")
        if first_newline != -1 and last_fence > first_newline:
            text = text[first_newline + 1:last_fence].strip()

    # Try parsing the whole text first
    result = _try_parse_json(text)
    if result:
        return result

    # Fallback: find the first { ... } JSON block in the text
    brace_start = text.find("{")
    while brace_start != -1:
        # Try from this { to end of text (allowing truncated JSON)
        candidate = text[brace_start:]
        result = _try_parse_json(candidate)
        if result:
            return result

        # Try finding a complete JSON block
        brace_count = 0
        for i in range(brace_start, len(text)):
            if text[i] == "{":
                brace_count += 1
            elif text[i] == "}":
                brace_count -= 1
            if brace_count == 0:
                candidate = text[brace_start:i + 1]
                result = _try_parse_json(candidate)
                if result:
                    return result
                break
        brace_start = text.find("{", brace_start + 1)

    return None


def _call_llm(prompt: str) -> Optional[Dict[str, Any]]:
    """Call LLM (mimo-v2.5-pro) and parse structured response.

    Returns parsed JSON dict on success, None on any failure.
    """
    api_key = os.environ.get("MIMO_API_KEY")
    if not api_key:
        print("[Growth Memory] MIMO_API_KEY not set, using template fallback")
        return None

    if OpenAI is None:
        print("[Growth Memory] openai package not installed, using template fallback")
        return None

    try:
        client = OpenAI(api_key=api_key, base_url=LLM_BASE_URL)
        print(f"[Growth Memory] Calling {LLM_MODEL}...")

        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "你是一位专业的婴幼儿安全顾问。请严格以 JSON 格式回复。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=8000,
        )

        msg = response.choices[0].message
        content = msg.content or ""

        # Reasoning models (like MiMo) may put output in reasoning_content
        if not content and hasattr(msg, "reasoning_content") and msg.reasoning_content:
            print(f"[Growth Memory] Found reasoning_content ({len(msg.reasoning_content)} chars)")
            content = msg.reasoning_content

        if not content:
            print("[Growth Memory] Empty response from LLM")
            return None

        result = _parse_llm_response(content)
        if result:
            print(f"[Growth Memory] LLM generated {len(result['memory_cards'])} cards, "
                  f"{len(result['parent_suggestions'])} suggestions")
            return result

        print("[Growth Memory] Failed to parse LLM response as JSON")
        print(f"[Growth Memory] Raw response (first 500 chars): {content[:500]}")
        return None

    except Exception as e:
        print(f"[Growth Memory] LLM API error: {e}")
        return None


# ── Node C: generate_insights ────────────────────────────────────────

def generate_insights_node(state: GrowthMemoryState) -> GrowthMemoryState:
    """Generate parent-facing memory cards, suggestions, and summary.

    Tries DeepSeek-V3 LLM first; falls back to built-in templates if
    the API is unavailable or returns invalid data.
    """
    log_step(state, "[Growth Memory] Generating insights...")

    event_count = state.get("event_count", 0)
    daily = state.get("daily_frequency", {})
    peak_hours = state.get("peak_hours", [])
    conf = state.get("confidence_trend", {})
    risk = state.get("risk_distribution", {})
    triggers = state.get("trigger_breakdown", {})
    positions = state.get("position_heatmap", {})
    date_range = state.get("date_range", {})

    # ── Pre-compute deterministic summary stats ──
    high_risk = risk.get("high", 0)
    num_days = len(daily) if daily else 1
    safety_score = max(0, min(100, 100 - (high_risk / max(num_days, 1)) * 3))
    near_edge = positions.get("near_crib_edge", 0)

    summary_stats = {
        "total_events": event_count,
        "days_covered": num_days,
        "avg_events_per_day": round(event_count / num_days, 1),
        "high_risk_count": high_risk,
        "peak_hours": peak_hours,
        "safety_score": round(safety_score),
        "confidence": conf,
    }

    # ── Try DeepSeek LLM for richer insights ──
    memory_cards: List[Dict[str, Any]] = []
    suggestions: List[Dict[str, Any]] = []

    prompt = _build_llm_prompt(state)
    llm_result = _call_llm(prompt)

    if llm_result is not None:
        # Use LLM-generated content
        memory_cards = llm_result["memory_cards"]
        suggestions = llm_result["parent_suggestions"]
        state["llm_used"] = True
        log_step(state, f"[Growth Memory] Used {LLM_MODEL} for insight generation")
    else:
        state["llm_used"] = False
        # ── Fallback: Built-in template logic ──
        log_step(state, "[Growth Memory] Using template fallback for insights")

        # Card 1: Exploration behavior trend
        if near_edge > 0:
            memory_cards.append({
                "title": "探索行为观察",
                "body": (
                    f"在 {date_range.get('start', '?')} 至 {date_range.get('end', '?')} 期间，"
                    f"宝宝共出现 {near_edge} 次靠近床沿或护栏边缘的行为。"
                    "这是一项短周期行为观察，提示床沿区域需要重点关注；"
                    "是否存在稳定变化仍需结合每日监控时长和更长周期数据验证。"
                ),
                "icon": "exploration",
                "severity": "info",
                "date": date_range.get("end", ""),
            })

        # Card 2: Risk pattern
        if high_risk > 0 and len(daily) > 1:
            avg_per_day = round(high_risk / len(daily), 1)
            if peak_hours:
                peak_str = "、".join(f"{h}点" for h in peak_hours[:2])
            else:
                peak_str = "部分时段"
            memory_cards.append({
                "title": "风险时段分析",
                "body": (
                    f"系统记录了 {high_risk} 次高风险事件，"
                    f"平均每天约 {avg_per_day} 次。"
                    f"高发时段集中在 {peak_str} 左右。"
                    f"建议在这些时段增加关注，或调整宝宝的活动安排。"
                ),
                "icon": "risk",
                "severity": "warning" if avg_per_day > 5 else "info",
                "date": date_range.get("end", ""),
            })

        # Card 3: Detection system maturity
        if triggers:
            main_trigger = max(triggers, key=triggers.get)
            main_count = triggers[main_trigger]
            total = sum(triggers.values())
            pct = round(main_count / total * 100)
            memory_cards.append({
                "title": "感知能力总结",
                "body": (
                    f"检测系统通过 {len(triggers)} 种方式识别宝宝活动，"
                    f"其中 {main_trigger} 贡献了 {pct}% 的事件触发量。"
                    "该结果用于说明当前 PoC 的事件来源构成，不代表真实环境准确率或产品可靠性。"
                ),
                "icon": "tech",
                "severity": "positive",
                "date": date_range.get("end", ""),
            })

        if not memory_cards:
            memory_cards.append({
                "title": "暂无成长记录",
                "body": "系统尚未记录到足够的活动数据。请先运行 Safety Agent 产生事件日志。",
                "icon": "empty",
                "severity": "info",
                "date": "",
            })

        # ── Template suggestions ──
        if near_edge > 10:
            suggestions.append({
                "title": "加强床沿防护",
                "body": "宝宝频繁靠近床沿，建议检查护栏高度是否足够，或考虑加装防撞软垫。",
                "priority": "high",
            })

        if peak_hours:
            h = peak_hours[0]
            if 6 <= h <= 9:
                suggestions.append({
                    "title": "晨间看护提醒",
                    "body": "宝宝在清晨时段活动量较大，建议这段时间安排家长在旁看护。",
                    "priority": "medium",
                })
            elif 18 <= h <= 22:
                suggestions.append({
                    "title": "晚间入睡前关注",
                    "body": "晚间时段是宝宝活动的高发期，建议在入睡前进行安全检查。",
                    "priority": "medium",
                })
            else:
                suggestions.append({
                    "title": "高发时段留意",
                    "body": f"宝宝在 {h} 点左右活动最为频繁，建议该时段保持关注。",
                    "priority": "medium",
                })

        if conf.get("average", 0) > 0.85:
            suggestions.append({
                "title": "继续验证检测效果",
                "body": "当前样本中的平均检测置信度较高，但置信度不等同于准确率或低误报率，仍需使用真实场景标注数据评估。",
                "priority": "low",
            })

        suggestions.append({
            "title": "定期检查设备",
            "body": "建议每周检查一次摄像头角度和婴儿床护栏状态，确保监测系统正常工作。",
            "priority": "low",
        })

    state["memory_cards"] = memory_cards
    state["parent_suggestions"] = suggestions
    state["summary_stats"] = summary_stats

    log_step(state, f"[Growth Memory] Generated {len(memory_cards)} memory cards, "
             f"{len(suggestions)} suggestions, safety_score={round(safety_score)}")
    return state


# ── Node D: render_output ────────────────────────────────────────────

def render_output_node(state: GrowthMemoryState) -> GrowthMemoryState:
    """Assemble final JSON output and write to disk."""
    log_step(state, "[Growth Memory] Rendering output...")

    date_range = state.get("date_range", {})
    output = {
        "generated_at": now_text(),
        "date_range": date_range,
        "event_count": state.get("event_count", 0),
        "llm_used": state.get("llm_used", False),
        "summary_stats": state.get("summary_stats", {}),
        "memory_cards": state.get("memory_cards", []),
        "parent_suggestions": state.get("parent_suggestions", []),
        "trend_data": {
            "daily_frequency": state.get("daily_frequency", {}),
            "hourly_distribution": {str(k): v for k, v in state.get("hourly_distribution", {}).items()},
            "peak_hours": state.get("peak_hours", []),
            "risk_distribution": state.get("risk_distribution", {}),
            "trigger_breakdown": state.get("trigger_breakdown", {}),
        },
    }

    state["output"] = output

    WEB_OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with WEB_OUTPUT_FILE.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    log_step(state, f"[Growth Memory] Written to {WEB_OUTPUT_FILE}")

    MEMORY_OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with MEMORY_OUTPUT_FILE.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    log_step(state, f"[Growth Memory] Written to {MEMORY_OUTPUT_FILE}")

    return state


# ── Graph Builder ─────────────────────────────────────────────────────

def build_growth_memory_graph():
    """Build the Growth Memory Agent LangGraph workflow."""
    workflow = StateGraph(GrowthMemoryState)

    workflow.add_node("load_events", load_events_node)
    workflow.add_node("analyze_trends", analyze_trends_node)
    workflow.add_node("generate_insights", generate_insights_node)
    workflow.add_node("render_output", render_output_node)

    workflow.set_entry_point("load_events")
    workflow.add_edge("load_events", "analyze_trends")
    workflow.add_edge("analyze_trends", "generate_insights")
    workflow.add_edge("generate_insights", "render_output")
    workflow.add_edge("render_output", END)

    return workflow.compile()


def main():
    print("=" * 72)
    print("Growth Memory Agent Demo")
    print("Reading event logs -> Analyzing trends -> Generating insights")
    print("=" * 72)
    print()

    app = build_growth_memory_graph()

    initial_state: GrowthMemoryState = {
        "events_file": str(SAMPLE_EVENTS_FILE),
        "logs": [],
    }

    final_state = app.invoke(initial_state)

    print()
    print("=" * 72)
    print("Growth Memory Agent Completed")
    print("=" * 72)

    output = final_state.get("output", {})
    print(f"\n  Events analyzed: {output.get('event_count', 0)}")
    print(f"  Memory cards:    {len(output.get('memory_cards', []))}")
    print(f"  Suggestions:     {len(output.get('parent_suggestions', []))}")
    print(f"  Safety score:    {output.get('summary_stats', {}).get('safety_score', 'N/A')}")
    print(f"\n  Output files:")
    print(f"    {WEB_OUTPUT_FILE}")
    print(f"    {MEMORY_OUTPUT_FILE}")


if __name__ == "__main__":
    main()
