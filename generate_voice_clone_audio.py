"""预生成语音克隆音频文件，用于 Web 演示。

支持两种 TTS 后端:
  1. MiMo TTS voiceclone（默认）— 无需预先克隆，每次调用传入参考音频
  2. Pipecat RemoteVoiceCloneService — 需要先克隆拿到 voice_id

用法:
    # MiMo TTS（默认，需 --api-key）
    python generate_voice_clone_audio.py --sample data/mom_sample.wav --api-key tp-xxx

    # Pipecat 服务（已有 voice_id）
    python generate_voice_clone_audio.py --provider pipecat --voice-id speech:xxx:xxx:hash
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import wave
from pathlib import Path

# ── 路径常量 ──────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent
AUDIO_DIR = PROJECT_ROOT / "web_demo" / "assets" / "audio"
GROWTH_MEMORY_JSON = PROJECT_ROOT / "web_demo" / "data" / "growth_memory.json"
MANIFEST_JSON = PROJECT_ROOT / "web_demo" / "data" / "voice_audio_manifest.json"

# ── 危险警告文本 ──────────────────────────────────────────

DANGER_WARNINGS = [
    {"filename": "danger_warning_1.wav", "text": "宝贝小心，请往中间来。"},
    {"filename": "danger_warning_2.wav", "text": "宝宝乖，不要靠近床边哦。"},
    {"filename": "danger_warning_3.wav", "text": "宝贝，妈妈在这里，慢慢往中间爬。"},
]


# ── MiMo TTS VoiceClone ──────────────────────────────────

def load_reference_audio_data_url(sample_path: str, duration_sec: int = 5) -> str:
    """读取参考音频并转为 DataURL（取前 N 秒）。"""
    p = Path(sample_path)
    if not p.exists():
        print(f"[错误] 参考音频不存在: {p}")
        sys.exit(1)

    with wave.open(str(p), "rb") as wf:
        sr = wf.getframerate()
        ch = wf.getnchannels()
        sw = wf.getsampwidth()
        frames = wf.readframes(sr * duration_sec)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as out:
        out.setnchannels(ch)
        out.setsampwidth(sw)
        out.setframerate(sr)
        out.writeframes(frames)

    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:audio/wav;base64,{b64}"


def mimo_tts_synthesize(
    text: str,
    voice_data_url: str,
    api_key: str,
    api_base: str = "https://token-plan-cn.xiaomimimo.com/v1",
) -> bytes:
    """调用 MiMo TTS voiceclone API，返回 WAV 字节。"""
    import ssl as _ssl
    import urllib.request

    ctx = _ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = _ssl.CERT_NONE

    payload = json.dumps({
        "model": "mimo-v2.5-tts-voiceclone",
        "messages": [{"role": "assistant", "content": text}],
        "audio": {"voice": voice_data_url, "format": "wav"},
        "stream": False,
    }).encode()

    req = urllib.request.Request(
        f"{api_base}/chat/completions",
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    resp = urllib.request.urlopen(req, context=ctx, timeout=120)
    result = json.loads(resp.read().decode())

    audio_b64 = result["choices"][0]["message"]["audio"]["data"]
    return base64.b64decode(audio_b64)


def resample_wav_to_48k(wav_path: Path) -> None:
    """将 WAV 文件重采样到 48kHz（MiMo 输出 24kHz，浏览器兼容性更好）。"""
    import array as _array

    with wave.open(str(wav_path), "rb") as wf:
        src_rate = wf.getframerate()
        if src_rate == 48000:
            return
        ch = wf.getnchannels()
        sw = wf.getsampwidth()
        frames = wf.readframes(wf.getnframes())

    samples = _array.array("h", frames)
    ratio = 48000 / src_rate
    new_count = int(len(samples) * ratio)
    resampled = _array.array("h", [0] * new_count)
    for i in range(new_count):
        pos = i / ratio
        idx = int(pos)
        frac = pos - idx
        if idx + 1 < len(samples):
            resampled[i] = int(samples[idx] * (1 - frac) + samples[idx + 1] * frac)
        else:
            resampled[i] = samples[idx]

    with wave.open(str(wav_path), "wb") as out:
        out.setnchannels(ch)
        out.setsampwidth(sw)
        out.setframerate(48000)
        out.writeframes(resampled.tobytes())


def generate_audio_mimo(
    voice_data_url: str,
    api_key: str,
    items: list[dict],
) -> dict[str, str]:
    """用 MiMo TTS 批量生成音频。"""
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    results: dict[str, str] = {}

    for item in items:
        filename = item["filename"]
        text = item["text"]
        out_path = AUDIO_DIR / filename

        print(f"[合成] {filename} — {text[:30]}...")
        try:
            audio_bytes = mimo_tts_synthesize(text, voice_data_url, api_key)
            out_path.write_bytes(audio_bytes)
            resample_wav_to_48k(out_path)
            rel_path = f"assets/audio/{filename}"
            results[filename] = rel_path
            print(f"  -> {out_path} ({len(audio_bytes)} bytes)")
        except Exception as exc:
            print(f"  [失败] {filename}: {exc}")

    return results


# ── Pipecat 远程克隆 ─────────────────────────────────────

def clone_voice_pipecat(sample_path: str, voice_id: str | None, api_base: str) -> str:
    """Pipecat 模式：克隆或使用已有 voice_id。"""
    sys.path.insert(0, str(PROJECT_ROOT / "Pipecat" / "Pipecat" / "src"))
    from services.remote_voice_clone_service import RemoteVoiceCloneService

    service = RemoteVoiceCloneService(api_base=api_base)

    if voice_id:
        print(f"[信息] 使用已有 voice_id: {voice_id}")
        return voice_id

    sample = Path(sample_path)
    if not sample.exists():
        print(f"[错误] 样本文件不存在: {sample}")
        sys.exit(1)

    print(f"[克隆] 上传样本: {sample.name}")
    result = service.clone_voice(
        sample_audio_path=sample,
        custom_name="baby_demo_mom",
        text="宝贝，妈妈在这里。",
    )
    new_voice_id = result.get("voice_id") or result.get("voice")
    if not new_voice_id:
        for v in result.values():
            if isinstance(v, str) and v.startswith("speech:"):
                new_voice_id = v
                break
    if not new_voice_id:
        print(f"[错误] 克隆返回中未找到 voice_id，原始返回: {result}")
        sys.exit(1)

    print(f"[克隆] 成功，voice_id: {new_voice_id}")
    return new_voice_id


def generate_audio_pipecat(
    voice_id: str,
    api_base: str,
    items: list[dict],
) -> dict[str, str]:
    """用 Pipecat RemoteVoiceCloneService 批量生成音频。"""
    sys.path.insert(0, str(PROJECT_ROOT / "Pipecat" / "Pipecat" / "src"))
    from services.remote_voice_clone_service import RemoteVoiceCloneService

    service = RemoteVoiceCloneService(api_base=api_base)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    results: dict[str, str] = {}

    for item in items:
        filename = item["filename"]
        text = item["text"]
        out_path = AUDIO_DIR / filename

        print(f"[合成] {filename} — {text[:30]}...")
        try:
            service.synthesize_to_file(text=text, voice=voice_id, output_path=out_path)
            rel_path = f"assets/audio/{filename}"
            results[filename] = rel_path
            print(f"  -> {out_path} ({out_path.stat().st_size} bytes)")
        except Exception as exc:
            print(f"  [失败] {filename}: {exc}")

    return results


# ── 公共逻辑 ─────────────────────────────────────────────

def load_growth_memory_texts() -> tuple[list[dict], list[dict]]:
    """从 growth_memory.json 提取记忆卡片和家长建议的文本。"""
    if not GROWTH_MEMORY_JSON.exists():
        print(f"[错误] 未找到 {GROWTH_MEMORY_JSON}")
        print("请先运行: python generate_growth_memory_web_data.py")
        sys.exit(1)

    data = json.loads(GROWTH_MEMORY_JSON.read_text(encoding="utf-8"))

    memory_texts = []
    for i, card in enumerate(data.get("memory_cards", []), start=1):
        memory_texts.append({
            "filename": f"memory_card_{i}.wav",
            "text": card["body"],
            "card_index": i - 1,
            "section": "memory_cards",
        })

    suggestion_texts = []
    for i, s in enumerate(data.get("parent_suggestions", []), start=1):
        suggestion_texts.append({
            "filename": f"suggestion_{i}.wav",
            "text": s["body"],
            "card_index": i - 1,
            "section": "parent_suggestions",
        })

    return memory_texts, suggestion_texts


def update_growth_memory_json(audio_map: dict[str, str], memory_texts: list[dict], suggestion_texts: list[dict]):
    """将 audio_file 字段写入 growth_memory.json。"""
    data = json.loads(GROWTH_MEMORY_JSON.read_text(encoding="utf-8"))

    for item in memory_texts:
        fn = item["filename"]
        idx = item["card_index"]
        if fn in audio_map and idx < len(data.get("memory_cards", [])):
            data["memory_cards"][idx]["audio_file"] = audio_map[fn]

    for item in suggestion_texts:
        fn = item["filename"]
        idx = item["card_index"]
        if fn in audio_map and idx < len(data.get("parent_suggestions", [])):
            data["parent_suggestions"][idx]["audio_file"] = audio_map[fn]

    GROWTH_MEMORY_JSON.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[更新] {GROWTH_MEMORY_JSON} — 已添加 audio_file 字段")


def write_manifest(danger_map: dict[str, str], provider: str, voice_id: str = ""):
    """写入危险警告音频映射清单。"""
    manifest = {
        "provider": provider,
        "voice_id": voice_id,
        "danger_warnings": [
            {"filename": w["filename"], "text": w["text"], "audio_file": danger_map.get(w["filename"], "")}
            for w in DANGER_WARNINGS
        ],
    }
    MANIFEST_JSON.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[写入] {MANIFEST_JSON}")


def main():
    parser = argparse.ArgumentParser(description="预生成语音克隆音频用于 Web 演示")
    parser.add_argument("--provider", choices=["mimo", "pipecat"], default="mimo", help="TTS 后端（默认 mimo）")
    parser.add_argument("--sample", help="语音样本文件路径（mimo: 参考音频; pipecat: 克隆样本）")
    parser.add_argument("--voice-id", help="Pipecat 模式：已有的克隆音色 ID")
    parser.add_argument("--api-key", default="", help="MiMo API Key（mimo 模式必填）")
    parser.add_argument("--api-base", default="https://token-plan-cn.xiaomimimo.com/v1", help="MiMo API 地址")
    parser.add_argument("--pipecat-api-base", default="http://223.247.96.246:30028", help="Pipecat API 地址")
    args = parser.parse_args()

    if args.provider == "mimo":
        if not args.sample:
            parser.error("mimo 模式需要 --sample 参考音频文件")
        if not args.api_key:
            parser.error("mimo 模式需要 --api-key")

        print(f"[MiMo] 加载参考音频: {args.sample}")
        voice_data_url = load_reference_audio_data_url(args.sample, duration_sec=5)
        print(f"[MiMo] DataURL 长度: {len(voice_data_url)} chars")

        memory_texts, suggestion_texts = load_growth_memory_texts()
        all_items = DANGER_WARNINGS + memory_texts + suggestion_texts

        print(f"\n{'='*50}")
        print(f"[合成] 共 {len(all_items)} 条音频（MiMo TTS voiceclone）")
        print(f"{'='*50}")

        all_map = generate_audio_mimo(voice_data_url, args.api_key, all_items)

        danger_map = {k: v for k, v in all_map.items() if k.startswith("danger_")}
        update_growth_memory_json(all_map, memory_texts, suggestion_texts)
        write_manifest(danger_map, provider="mimo")

    else:  # pipecat
        if not args.voice_id and not args.sample:
            parser.error("pipecat 模式需要 --voice-id 或 --sample")

        voice_id = clone_voice_pipecat(args.sample, args.voice_id, args.pipecat_api_base)

        memory_texts, suggestion_texts = load_growth_memory_texts()
        all_items = DANGER_WARNINGS + memory_texts + suggestion_texts

        print(f"\n{'='*50}")
        print(f"[合成] 共 {len(all_items)} 条音频（Pipecat RemoteVoiceClone）")
        print(f"{'='*50}")

        all_map = generate_audio_pipecat(voice_id, args.pipecat_api_base, all_items)

        danger_map = {k: v for k, v in all_map.items() if k.startswith("danger_")}
        update_growth_memory_json(all_map, memory_texts, suggestion_texts)
        write_manifest(danger_map, provider="pipecat", voice_id=voice_id)

    # 汇总
    total = len(DANGER_WARNINGS) + len(memory_texts) + len(suggestion_texts)
    print(f"\n{'='*50}")
    print(f"[完成] 生成 {len(all_map)}/{total} 个音频文件")
    print(f"  输出目录: {AUDIO_DIR}")
    if len(all_map) < total:
        print("  [警告] 部分文件生成失败，请检查上方日志")
    print(f"\n下一步：")
    print(f"  cd web_demo && python start_server.py")
    print(f"  打开 http://localhost:8080 测试语音播放")


if __name__ == "__main__":
    main()
