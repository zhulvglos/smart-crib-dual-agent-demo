"""
启动网页端Demo的HTTP服务器
支持局域网访问（手机、其他设备可通过IP访问）
"""

import mimetypes
import socket
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PORT = 8080
WEB_DEMO_DIR = Path(__file__).resolve().parent


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
            self.send_response(200)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(size))
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            with open(file_path, 'rb') as f:
                self.wfile.write(f.read())
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
