import socket
import threading
import json
import os
import cv2
import time
import traceback

from server_config import ServerConfig, AppState
from vision_engine import VisionEngine


class HTTPRequest:
    def __init__(self, raw_data):
        self.method = ""
        self.path = ""
        self.headers = {}
        self.body = b""
        self._parse(raw_data)

    def _parse(self, raw_data):
        header_part, _, self.body = raw_data.partition(b'\r\n\r\n')
        lines = header_part.decode('utf-8', errors='ignore').split('\r\n')

        if lines:
            request_line = lines[0].split(' ')

            if len(request_line) >= 2:
                self.method = request_line[0]
                self.path = request_line[1]

            for line in lines[1:]:
                if ': ' in line:
                    parts = line.split(': ', 1)

                    if len(parts) == 2:
                        self.headers[parts[0].strip()] = parts[1].strip()


class HTTPResponse:
    def __init__(self, status_code=200, content_type="text/html", body=b""):
        self.status_code = status_code
        self.content_type = content_type
        self.body = body if isinstance(body, bytes) else body.encode()

        self.headers = {
            "Content-Type": self.content_type,
            "Content-Length": len(self.body),
            "Access-Control-Allow-Origin": "*",
            "Connection": "close"
        }

    def to_bytes(self):
        status_text = {
            200: "OK",
            201: "Created",
            400: "Bad Request",
            403: "Forbidden",
            404: "Not Found",
            500: "Internal Server Error"
        }.get(self.status_code, "Unknown")

        self.headers["Content-Length"] = len(self.body)

        resp = f"HTTP/1.1 {self.status_code} {status_text}\r\n"

        for k, v in self.headers.items():
            resp += f"{k}: {v}\r\n"

        resp += "\r\n"

        return resp.encode() + self.body


class RawHTTPServer:
    def __init__(self, config: ServerConfig, state: AppState, vision: VisionEngine):
        self.config = config
        self.state = state
        self.vision = vision
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    def start(self):
        self.server_socket.bind((self.config.HOST, self.config.PORT))
        self.server_socket.listen(10)

        print(f"[*] Server listening on http://{self.config.HOST}:{self.config.PORT}")

        try:
            while True:
                client_sock, addr = self.server_socket.accept()

                threading.Thread(
                    target=self.handle_client,
                    args=(client_sock, addr),
                    daemon=True
                ).start()

        except KeyboardInterrupt:
            self.stop()

        except Exception as e:
            print("[SERVER ERROR]", e)
            traceback.print_exc()
            self.stop()

    def stop(self):
        print("\n[*] Shutting down...")

        try:
            self.vision.release()
        except Exception as e:
            print("[VISION RELEASE ERROR]", e)

        try:
            self.server_socket.close()
        except Exception as e:
            print("[SERVER SOCKET CLOSE ERROR]", e)

    def log_request(self, request, addr=None):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        if addr:
            client_ip, client_port = addr
            print(f"[{timestamp}] {client_ip}:{client_port} -> {request.method} {request.path}")
        else:
            print(f"[{timestamp}] {request.method} {request.path}")

    def handle_client(self, client_sock, addr=None):
        client_sock.settimeout(5.0)

        try:
            data = b""

            while b'\r\n\r\n' not in data:
                chunk = client_sock.recv(2048)

                if not chunk:
                    break

                data += chunk

            if not data:
                return

            request = HTTPRequest(data)

            self.log_request(request, addr)

            if request.method == "GET" and request.path == "/stream":
                self._handle_stream(client_sock)
                return

            if "Content-Length" in request.headers:
                content_len = int(request.headers["Content-Length"])

                while len(request.body) < content_len:
                    chunk = client_sock.recv(4096)

                    if not chunk:
                        break

                    request.body += chunk

            response = self.route(request)

            if response:
                if request.method == "HEAD":
                    response.body = b""

                client_sock.sendall(response.to_bytes())

        except Exception as e:
            print("[INTERNAL ERROR]", e)
            traceback.print_exc()

            try:
                response = HTTPResponse(500, body="Internal Server Error")
                client_sock.sendall(response.to_bytes())
            except Exception as send_error:
                print("[500 RESPONSE SEND ERROR]", send_error)

        finally:
            try:
                client_sock.close()
            except Exception as close_error:
                print("[CLIENT SOCKET CLOSE ERROR]", close_error)

    def route(self, request):
        if request.method in ["GET", "HEAD"]:
            if request.path == "/" or request.path == "/index.html":
                return self.serve_static("index.html")

            if request.path.startswith("/assets/"):
                return self.serve_static(request.path.lstrip("/"))

            if request.path == "/models":
                return self._handle_list_models()

        if request.method == "PUT" and request.path == "/state":
            return self._handle_update_state(request)

        if request.method == "POST":
            if request.path == "/record/start":
                filename = self.vision.start_recording()
                if filename:
                    return HTTPResponse(200, content_type="application/json", body=json.dumps({"status": "recording_started", "filename": filename}))
                else:
                    return HTTPResponse(400, body=json.dumps({"status": "already_recording"}))
            elif request.path == "/record/stop":
                self.vision.stop_recording()
                return HTTPResponse(200, content_type="application/json", body=json.dumps({"status": "recording_stopped"}))

        return HTTPResponse(404, body="Not Found")

    def serve_static(self, filename):
        path = filename if os.path.exists(filename) else os.path.join(self.config.STATIC_DIR, filename)

        if not os.path.exists(path) or os.path.isdir(path):
            return HTTPResponse(404, body="File not found")

        ext = os.path.splitext(filename)[1].lower()

        content_type = {
            ".html": "text/html",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".css": "text/css",
            ".js": "application/javascript",
            ".json": "application/json",
            ".ico": "image/x-icon",
            ".gif": "image/gif"
        }.get(ext, "text/plain")

        with open(path, 'rb') as f:
            content = f.read()

        return HTTPResponse(200, content_type=content_type, body=content)

    def _handle_list_models(self):
        models = self.config.get_available_models()

        return HTTPResponse(
            200,
            content_type="application/json",
            body=json.dumps({"models": models})
        )

    def _handle_stream(self, client_sock):
        header = (
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: multipart/x-mixed-replace; boundary=frame\r\n"
            "Access-Control-Allow-Origin: *\r\n"
            "Connection: close\r\n"
            "\r\n"
        )

        try:
            client_sock.sendall(header.encode())

            while True:
                frame = self.vision.get_host_frame()

                if frame is None:
                    time.sleep(0.01)
                    continue

                success, buffer = cv2.imencode('.jpg', frame)

                if not success:
                    print("[STREAM ERROR] Failed to encode JPEG frame")
                    continue

                frame_bytes = buffer.tobytes()

                content = (
                    f"--frame\r\n"
                    f"Content-Type: image/jpeg\r\n"
                    f"Content-Length: {len(frame_bytes)}\r\n"
                    f"\r\n"
                ).encode() + frame_bytes + b"\r\n"

                client_sock.sendall(content)
                time.sleep(0.04)

        except BrokenPipeError:
            print("[STREAM] Client disconnected")

        except ConnectionResetError:
            print("[STREAM] Connection reset by client")

        except Exception as e:
            print("[STREAM ERROR]", e)
            traceback.print_exc()

    def _handle_update_state(self, request):
        try:
            data = json.loads(request.body.decode())

            if 'model' in data:
                self.state.active_model = data['model']
                self.vision._load_model()

            return HTTPResponse(
                200,
                content_type="application/json",
                body=json.dumps({"status": "updated"})
            )

        except json.JSONDecodeError:
            return HTTPResponse(400, body="Invalid JSON")

        except Exception as e:
            print("[STATE UPDATE ERROR]", e)
            traceback.print_exc()

            return HTTPResponse(500, body="Internal Server Error")


if __name__ == "__main__":
    config = ServerConfig()
    state = AppState(config)
    vision = VisionEngine(state)

    server = RawHTTPServer(config, state, vision)
    server.start()
