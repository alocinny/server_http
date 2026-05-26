import os
from dataclasses import dataclass, field

@dataclass
class ServerConfig:
    HOST: str = '0.0.0.0'
    PORT: int = 8080
    MODELS_DIR: str = 'models'
    STATIC_DIR: str = 'static'
    MEDIA_DIR: str = 'media'
    CONFIDENCE_THRESHOLD: float = 0.5
    
    def get_available_models(self):
        if not os.path.exists(self.MODELS_DIR):
            return []
        # Return all .pt and .onnx files in the models directory
        return [f for f in os.listdir(self.MODELS_DIR) 
                if f.endswith(('.pt', '.onnx'))]

class AppState:
    def __init__(self, config: ServerConfig):
        self.config = config
        models = self.config.get_available_models()
        self.active_model = models[0] if models else ""
        self.camera_source = 'host'
        self.is_recording = False
        
        os.makedirs(config.MEDIA_DIR, exist_ok=True)
        os.makedirs(config.STATIC_DIR, exist_ok=True)
        os.makedirs(config.MODELS_DIR, exist_ok=True)
