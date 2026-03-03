import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Initialize MediaPipe Tasks Pose Landmarker
model_dir = os.path.dirname(__file__)
# You might need to download 'pose_landmarker_heavy.task' or similar. 
# For now, we'll implement a fallback if the task file isn't found.
# Since we only need human body gating, we can also use a simpler technique 
# or try to re-install a version of mediapipe that has solutions if preferred.

# However, to be fast and avoid another 100MB download during this chat:
# Let's use a standard OpenCV-based HOG descriptor for human detection 
# which is built-in and very reliable for "Is there a person here?" gating.

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

class ModelHandler:
    def __init__(self, model_path):
        try:
            import tflite_runtime.interpreter as tflite
        except ImportError:
            import tensorflow.lite as tflite
            
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Get input shape from model
        self.input_shape = self.input_details[0]['shape']
        self.input_height = self.input_shape[1]
        self.input_width = self.input_shape[2]
        
    def is_human_body_present(self, img_bgr):
        """Use OpenCV HOG + SVM to verify if a human body is present."""
        # Gray scale for faster detection
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # Detect people in the image
        # winStride: step size in pixels for the sliding window
        # padding: pixels added to the ROI before detection
        # scale: scale factor between images in the pyramid
        (rects, weights) = hog.detectMultiScale(gray, winStride=(8, 8), padding=(8, 8), scale=1.05)
        
        return len(rects) > 0 # At least one person detected

    def preprocess_image(self, img):
        # 1. Resize to model input size (e.g., 224x224)
        frame_resized = cv2.resize(img, (self.input_width, self.input_height))
        
        # 2. Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        
        # 3. Add batch dimension
        input_data = np.expand_dims(frame_rgb, axis=0)
        
        # 4. Convert to the required type
        dtype = self.input_details[0]['dtype']
        if dtype == np.float32:
            input_data = input_data.astype(np.float32) / 255.0
        elif dtype == np.uint8:
            input_data = input_data.astype(np.uint8)
            
        return input_data

    def predict(self, image_bytes):
        try:
            # Convert bytes to OpenCV format
            nparr = np.frombuffer(image_bytes, np.uint8)
            img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img_bgr is None:
                raise ValueError("Failed to decode image")

            # 1. Run the TFLite model directly
            # Note: We removed the HOG gate because it often fails during active falling/motion
            input_data = self.preprocess_image(img_bgr)
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()
            
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
            pred_class = np.argmax(output_data)
            confidence = float(output_data[pred_class])
            
            # STRICT LOGIC: 
            # Class 0 = FALL
            # Class 1 = NORMAL
            CONF_THRESH = 0.80 
            is_fall = bool(pred_class == 0 and confidence > CONF_THRESH)
            
            return {
                "is_fall": is_fall, 
                "confidence": int(confidence * 100),
                "class_id": int(pred_class),
                "raw_confidence": confidence,
                "status": "Fall Detected" if is_fall else "Normal"
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
