import os
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    try:
        import tensorflow.lite as tflite
    except ImportError:
        raise ImportError("Neither 'tflite-runtime' nor 'tensorflow' is installed.")

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=0, min_detection_confidence=0.5)

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
        
    def is_human_body_present(self, img_rgb):
        """Use MediaPipe to verify if a human body is present."""
        results = pose.process(img_rgb)
        if not results.pose_landmarks:
            return False
            
        # Optional: Check for key landmarks like shoulders or hips
        # (lm 11, 12, 23, 24 are shoulders and hips)
        # If we only see a face (lm 0-10), MediaPipe might still return true,
        # so we can check if landmarks below the face are visible.
        landmarks = results.pose_landmarks.landmark
        body_parts = [11, 12, 23, 24] # Shoulders and Hips
        visible_parts = [lm for i, lm in enumerate(landmarks) if i in body_parts and lm.visibility > 0.5]
        
        return len(visible_parts) >= 2 # At least 2 body parts must be visible

    def preprocess_image(self, img):
        # 1. Resize to model input size (e.g., 224x224)
        frame_resized = cv2.resize(img, (self.input_width, self.input_height))
        
        # 2. Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        
        # 3. Normalize: Convert to float32 and scale to [0, 1]
        input_data = np.expand_dims(frame_rgb, axis=0).astype(np.float32) / 255.0
            
        return input_data

    def predict(self, image_bytes):
        try:
            # Convert bytes to OpenCV format
            nparr = np.frombuffer(image_bytes, np.uint8)
            img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img_bgr is None:
                raise ValueError("Failed to decode image")

            # First, check if a human body is present using MediaPipe
            img_rgb_full = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            if not self.is_human_body_present(img_rgb_full):
                return {"is_fall": False, "confidence": 0, "status": "No human body detected"}

            # If body is present, run the TFLite model
            input_data = self.preprocess_image(img_bgr)
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()
            
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
            pred_class = np.argmax(output_data)
            confidence = float(output_data[pred_class])
            
            # Use the requested strict threshold (0.9)
            CONF_THRESH = 0.90 
            is_fall = (pred_class == 1 and confidence > CONF_THRESH)
            
            return {
                "is_fall": is_fall, 
                "confidence": int(confidence * 100),
                "class_id": int(pred_class),
                "raw_confidence": confidence,
                "status": "Body detected"
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
