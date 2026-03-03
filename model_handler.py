import os
import cv2
import numpy as np
from PIL import Image

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    try:
        import tensorflow.lite as tflite
    except ImportError:
        raise ImportError("Neither 'tflite-runtime' nor 'tensorflow' is installed.")

class ModelHandler:
    def __init__(self, model_path):
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Get input shape from model
        self.input_shape = self.input_details[0]['shape']
        self.input_height = self.input_shape[1]
        self.input_width = self.input_shape[2]
        
    def preprocess_image(self, image_bytes):
        # Convert bytes to OpenCV format
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image")
            
        # 1. Resize to model input size (e.g., 224x224)
        frame_resized = cv2.resize(img, (self.input_width, self.input_height))
        
        # 2. Convert BGR to RGB (OpenCV uses BGR by default)
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        
        # 3. Normalize: Convert to float32 and scale to [0, 1]
        input_data = np.expand_dims(frame_rgb, axis=0).astype(np.float32) / 255.0
            
        return input_data

    def predict(self, image_bytes):
        try:
            input_data = self.preprocess_image(image_bytes)
            
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()
            
            # Get raw model output
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
            
            # Use argmax to find the predicted class
            pred_class = np.argmax(output_data)
            confidence = float(output_data[pred_class])
            
            # Logic from your snippet: Assume Class 1 is 'Fall'
            # and use your requested threshold (e.g., 0.8)
            # We keep it at 0.9 for extra stability as requested earlier
            STRICT_THRESHOLD = 0.90 
            
            is_fall = (pred_class == 1 and confidence > STRICT_THRESHOLD)
            
            return {
                "is_fall": is_fall, 
                "confidence": int(confidence * 100),
                "class_id": int(pred_class),
                "raw_confidence": confidence
            }
        except Exception as e:
            print(f"Prediction error: {e}")
            return {"is_fall": False, "confidence": 0, "error": str(e)}

# Initialize model handler
model_path = os.path.join(os.path.dirname(__file__), "../Fall_Detection/tflite-model-maker-falldetect-model.tflite")
handler = None
if os.path.exists(model_path):
    try:
        handler = ModelHandler(model_path)
    except Exception as e:
        print(f"Failed to initialize model: {e}")
else:
    print(f"Model file not found at: {model_path}")
