"""
启动网页端Demo的HTTP服务器
支持局域网访问（手机、其他设备可通过IP访问）
"""

import mimetypes
import json
import socket
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PORT = 8080
WEB_DEMO_DIR = Path(__file__).resolve().parent
PROJECT_DIR = WEB_DEMO_DIR.parent
VOICE_EVENT_LOG = PROJECT_DIR / "logs" / "voice_companion_events.jsonl"


def get_local_ip():
    """获取本机局域网IP地址"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


class DemoHandler(BaseHTTPRequestHandler):
    def send_json(self, status, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_POST(self):
        path = self.path.split("?")[0]
        if path != "/api/voice-companion/result":
            self.send_json(404, {"ok": False, "error": "Unknown API endpoint"})
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length <= 0 or content_length > 32 * 1024:
                self.send_json(400, {"ok": False, "error": "Invalid request size"})
                return

            payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
            required = {
                "event_id",
                "event_type",
                "scene",
                "selected_voice",
                "selected_voice_label",
                "comfort_script",
                "background_audio",
                "selection_reason",
                "is_simulated",
                "outcome",
            }
            missing = sorted(required - payload.keys())
            if missing:
                self.send_json(
                    400,
                    {"ok": False, "error": f"Missing fields: {', '.join(missing)}"},
                )
                return
            if payload.get("event_type") != "crying_comfort":
                self.send_json(400, {"ok": False, "error": "Invalid event type"})
                return
            if payload.get("is_simulated") is not True:
                self.send_json(400, {"ok": False, "error": "Demo events must be simulated"})
                return

            record = {
                key: payload[key]
                for key in (
                    "event_id",
                    "event_type",
                    "scene",
                    "selected_voice",
                    "selected_voice_label",
                    "comfort_script",
                    "background_audio",
                    "selection_reason",
                    "is_simulated",
                    "outcome",
                )
            }
            record["timestamp"] = payload.get("timestamp") or datetime.now().astimezone().isoformat(timespec="seconds")
            if payload.get("baby_id"):
                record["baby_id"] = payload["baby_id"]
            if payload.get("source_event_id"):
                record["source_event_id"] = payload["source_event_id"]

            VOICE_EVENT_LOG.parent.mkdir(parents=True, exist_ok=True)
            with VOICE_EVENT_LOG.open("a", encoding="utf-8") as file:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")

            self.send_json(
                200,
                {
                    "ok": True,
                    "message": "模拟安抚记录已写入 JSONL",
                    "log_file": "logs/voice_companion_events.jsonl",
                    "record": record,
                },
            )
        except json.JSONDecodeError:
            self.send_json(400, {"ok": False, "error": "Invalid JSON"})
        except Exception as error:
            self.send_json(500, {"ok": False, "error": str(error)})

    def do_GET(self):
        path = self.path.split('?')[0].lstrip('/')
        file_path = (WEB_DEMO_DIR / path).resolve()

        try:
            file_path.relative_to(WEB_DEMO_DIR)
        except ValueError:
            self.send_error(403, "Forbidden")
            return

        if file_path.is_dir():
            index = file_path / 'index.html'
            if index.is_file():
                file_path = index
            else:
                self.send_error(404, "Not Found")
                return

        if not file_path.is_file():
            self.send_error(404, "Not Found: " + path)
            return

        content_type, _ = mimetypes.guess_type(str(file_path))
        if content_type is None:
            content_type = 'application/octet-stream'

        try:
            size = file_path.stat().st_size
            range_header = self.headers.get("Range")
            start = 0
            end = size - 1
            status = 200

            if range_header and range_header.startswith("bytes="):
                byte_range = range_header[len("bytes="):].split(",", 1)[0]
                start_text, end_text = byte_range.split("-", 1)
                if start_text:
                    start = int(start_text)
                if end_text:
                    end = min(int(end_text), size - 1)
                if start >= size or start > end:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.end_headers()
                    return
                status = 206

            content_length = end - start + 1
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Accept-Ranges', 'bytes')
            self.send_header('Content-Length', str(content_length))
            if status == 206:
                self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            with open(file_path, 'rb') as f:
                f.seek(start)
                remaining = content_length
                while remaining > 0:
                    chunk = f.read(min(64 * 1024, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        except BrokenPipeError:
            pass
        except Exception as e:
            self.send_error(500, str(e))

    def log_message(self, format, *args):
        pass


if __name__ == "__main__":
    local_ip = get_local_ip()

    ThreadingHTTPServer.allow_reuse_address = True
    server = ThreadingHTTPServer(("0.0.0.0", PORT), DemoHandler)

    print()
    print("=" * 55)
    print("  AI婴儿床监护系统 - 网页端Demo")
    print("=" * 55)
    print()
    print("  本机访问:   http://localhost:" + str(PORT) + "/index.html")
    print("  局域网访问: http://" + local_ip + ":" + str(PORT) + "/index.html")
    print()
    print("  手机访问步骤:")
    print("    1. 手机连接同一个WiFi")
    print("    2. 浏览器打开上面的「局域网访问」地址")
    print("    3. 如果打不开，需要在Windows防火墙中允许Python")
    print()
    print("  转发给朋友:")
    print("    - 局域网内: 直接发上面的「局域网访问」地址")
    print("    - 外网: 需要使用内网穿透工具(如ngrok)")
    print()
    print("  Ctrl+C 停止服务器")
    print("=" * 55)
    print()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")
    finally:
        server.server_close()
