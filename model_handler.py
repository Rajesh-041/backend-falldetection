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
        raise ImportError("Neither 'tflite-runtime' nor 'tensorflow' is installed. Please install one of them.")

class ModelHandler:
    def __init__(self, model_path):
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Get input shape
        self.input_shape = self.input_details[0]['shape']
        self.input_height = self.input_shape[1]
        self.input_width = self.input_shape[2]
        
    def preprocess_image(self, image_bytes):
        # Convert bytes to PIL Image
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize image for model input
        img_resized = cv2.resize(img_rgb, (self.input_width, self.input_height))
        
        # Normalize image if needed (assuming 0-255 to 0-1)
        # Check input type
        if self.input_details[0]['dtype'] == np.float32:
            input_data = np.expand_dims(img_resized, axis=0).astype(np.float32) / 255.0
        else:
            input_data = np.expand_dims(img_resized, axis=0).astype(np.uint8)
            
        return input_data

    def predict(self, image_bytes):
        try:
            input_data = self.preprocess_image(image_bytes)
            
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()
            
            # Get prediction result
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
            
            # --- DRASTIC CONFIDENCE REDUCTION ---
            # Increase threshold from 0.5 to 0.95 (95% certainty required)
            STRICT_THRESHOLD = 0.95

            if len(output_data) >= 2:
                # If it's [prob_not_fall, prob_fall]
                raw_confidence = float(output_data[1])
            else:
                # If it's a single value [prob_fall]
                raw_confidence = float(output_data[0])

            # Apply a damping factor to the reported confidence (scale it down)
            # This makes the model "humble" and less likely to hit the threshold
            damped_confidence = raw_confidence * 0.8 

            is_fall = raw_confidence > STRICT_THRESHOLD

            return {
                "is_fall": is_fall, 
                "confidence": int(damped_confidence * 100),
                "raw_val": float(raw_confidence)
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
