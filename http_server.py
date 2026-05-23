import socket
import threading
import json
import os
import cv2
import time
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
        status_text = {200: "OK", 201: "Created", 400: "Bad Request", 404: "Not Found"}.get(self.status_code, "Unknown")
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
                threading.Thread(target=self.handle_client, args=(client_sock,), daemon=True).start()
        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        print("\n[*] Shutting down...")
        self.vision.release()
        self.server_socket.close()

    def handle_client(self, client_sock):
        client_sock.settimeout(5.0)
        try:
            data = b""
            while b'\r\n\r\n' not in data:
                chunk = client_sock.recv(2048)
                if not chunk: break
                data += chunk
            if not data: return
            request = HTTPRequest(data)
            if request.method == "GET" and request.path == "/stream":
                self._handle_stream(client_sock)
                return
            if "Content-Length" in request.headers:
                try:
                    content_len = int(request.headers["Content-Length"])
                    while len(request.body) < content_len:
                        chunk = client_sock.recv(4096)
                        if not chunk: break
                        request.body += chunk
                except: pass
            response = self.route(request)
            if response:
                client_sock.sendall(response.to_bytes())
        except: pass
        finally:
            try: client_sock.close()
            except: pass

    def route(self, request):
        if request.method == "GET":
            if request.path == "/" or request.path == "/index.html": return self.serve_static("index.html")
            if request.path == "/models": return self._handle_list_models()
        if request.method == "PUT" and request.path == "/state":
            return self._handle_update_state(request)
        return HTTPResponse(404, body="Not Found")

    def serve_static(self, filename):
        path = filename if os.path.exists(filename) else os.path.join(self.config.STATIC_DIR, filename)
        if not os.path.exists(path): return HTTPResponse(404, body="File not found")
        with open(path, 'rb') as f: content = f.read()
        return HTTPResponse(200, content_type="text/html", body=content)

    def _handle_list_models(self):
        models = self.config.get_available_models()
        return HTTPResponse(200, content_type="application/json", body=json.dumps({"models": models}))

    def _handle_stream(self, client_sock):
        header = "HTTP/1.1 200 OK\r\nContent-Type: multipart/x-mixed-replace; boundary=frame\r\nAccess-Control-Allow-Origin: *\r\n\r\n"
        try:
            client_sock.sendall(header.encode())
            while True:
                frame = self.vision.get_host_frame()
                if frame is None:
                    time.sleep(0.01)
                    continue
                _, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()
                content = (f"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: {len(frame_bytes)}\r\n\r\n").encode() + frame_bytes + b"\r\n"
                client_sock.sendall(content)
                time.sleep(0.04)
        except: pass

    def _handle_update_state(self, request):
        try:
            data = json.loads(request.body.decode())
            if 'model' in data:
                self.state.active_model = data['model']
                self.vision._load_model()
            return HTTPResponse(200, body=json.dumps({"status": "updated"}))
        except: return HTTPResponse(400, body="Invalid data")

if __name__ == "__main__":
    config = ServerConfig()
    state = AppState(config)
    vision = VisionEngine(state)
    server = RawHTTPServer(config, state, vision)
    server.start()