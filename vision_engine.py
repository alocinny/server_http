import cv2
import os
import time
import numpy as np
import threading
from server_config import AppState

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

class VisionEngine:
    def __init__(self, state: AppState):
        self.state = state
        self.host_cap = None
        self.model = None
        self.video_writer = None
        self.is_recording = False
        self.video_path = None
        self.video_fps = 30.0
        self.current_frame = None
        self.running = True
        self.lock = threading.Lock()
        self._load_model() 

        self.capture_thread = threading.Thread(target=self._capture_loop,
                                               daemon=True)
        self.capture_thread.start()

    def _load_model(self):
        model_path = os.path.join(self.state.config.MODELS_DIR, self.state.active_model)
        if not os.path.exists(model_path):
            print(f"[!] Model file not found: {model_path}")
            return

        try:
            if self.state.active_model.endswith('.pt') and ULTRALYTICS_AVAILABLE:
                self.model = YOLO(model_path)
                print(f"[*] YOLO .pt model {self.state.active_model} loaded.")
            else:
                self.model = cv2.dnn.readNet(model_path)
                self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                print(f"[*] Model {self.state.active_model} loaded via OpenCV.")
        except Exception as e:
            print(f"[!] Error loading model: {e}")
            self.model = None

    def process_frame(self, frame):
        if frame is None or self.model is None:
            return frame

        h, w = frame.shape[:2]
        threshold = self.state.config.CONFIDENCE_THRESHOLD
        
        # .pt model logic
        if self.state.active_model.endswith('.pt') and ULTRALYTICS_AVAILABLE:
            try:
                results = self.model(frame, verbose=False)[0]
                for box in results.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    name = self.model.names[cls_id]
                    
                    if conf > threshold:
                        label = f"{name} {conf:.2f}"
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        cv2.putText(frame, label, (int(x1), int(y1) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            except Exception as e:
                print(f"[!] .pt processing error: {e}")

        # .onnx fallback logic
        elif isinstance(self.model, cv2.dnn.Net):
            try:
                blob = cv2.dnn.blobFromImage(frame, 1/255.0, (640, 640), swapRB=True, crop=False)
                self.model.setInput(blob)
                outputs = self.model.forward(self.model.getUnconnectedOutLayersNames())
                predictions = np.squeeze(outputs[0]).T
                x_factor, y_factor = w / 640, h / 640

                if predictions.ndim == 2:
                    for pred in predictions:
                        scores = pred[4:]
                        conf = scores.max()
                        if conf > threshold:
                            cx, cy, bw, bh = pred[:4]
                            left, top = int((cx-bw/2)*x_factor), int((cy-bh/2)*y_factor)
                            cv2.rectangle(frame, (left, top), (left+int(bw*x_factor), top+int(bh*y_factor)), (0, 255, 0), 2)
            except: pass

        return frame

    def get_host_frame(self):
        with self.lock:
            if self.current_frame is not None:
                return self.current_frame.copy()
        return None
    def stop_host_camera(self):
        if self.host_cap:
            self.host_cap.release()
            self.host_cap = None

    def save_media(self, frame, filename=None):
        if frame is None: return None
        if filename is None:
            filename = f"snap_{int(time.time())}.jpg"
        path = os.path.join(self.state.config.MEDIA_DIR, filename)
        cv2.imwrite(path, frame)
        return filename

    def start_recording(self, fps=None):
        with self.lock:
            if self.is_recording: return None
            
            filename = f"video_{int(time.time())}.mp4"
            self.video_path = os.path.join(self.state.config.MEDIA_DIR, filename)
            # Use dynamically tracked FPS if fps is not provided
            self.video_fps = fps if fps is not None else getattr(self, 'actual_fps', 20.0)
            self.is_recording = True
            
            print(f"[*] Gravação solicitada: {filename} a {self.video_fps:.1f} FPS")
            return filename

    def stop_recording(self):
        with self.lock:
            if self.is_recording:
                if self.video_writer is not None:
                    self.video_writer.release()
                    self.video_writer = None
                self.is_recording = False
                self.video_path = None
                print("[*] Gravação finalizada de forma segura.")
                
    def _capture_loop(self):
        self.host_cap = cv2.VideoCapture(0)
        
        last_time = time.time()
        self.actual_fps = 20.0
        fps_alpha = 0.05
        
        while self.running:
            ret, frame = self.host_cap.read()
            if not ret:
                time.sleep(0.01)
                continue
                
            processed_frame = self.process_frame(frame)
            
            current_time = time.time()
            dt = current_time - last_time
            last_time = current_time
            
            if dt > 0:
                loop_fps = 1.0 / dt
                self.actual_fps = (1 - fps_alpha) * self.actual_fps + fps_alpha * loop_fps
            
            with self.lock:
                self.current_frame = processed_frame.copy() if processed_frame is not None else None
                
                if self.is_recording:
                    if self.video_writer is None and self.video_path is not None:
                        h, w = processed_frame.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        safe_fps = max(5.0, self.video_fps)
                        self.video_writer = cv2.VideoWriter(self.video_path, fourcc, safe_fps, (w, h))

                    if self.video_writer is not None:
                        self.video_writer.write(processed_frame)
                    
            time.sleep(0.01)
            
        if self.host_cap:
            self.host_cap.release()

    def release(self):
        self.running = False
        self.stop_recording()
        if hasattr(self, 'capture_thread') and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=1.0)
