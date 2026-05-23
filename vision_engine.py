import cv2
import os
import time
import numpy as np
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
        self._load_model()

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

        cv2.putText(frame, f"Model: {self.state.active_model}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        return frame

    def get_host_frame(self):
        if self.host_cap is None or not self.host_cap.isOpened():
            self.host_cap = cv2.VideoCapture(0)
        ret, frame = self.host_cap.read()
        if not ret: return None
        return self.process_frame(frame)

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

    def release(self):
        self.stop_host_camera()