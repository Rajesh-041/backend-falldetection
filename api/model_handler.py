import os
import cv2
import numpy as np

# -------- LOAD TFLITE --------
try:
    # New LiteRT (supports Python 3.12+)
    import ai_edge_litert.interpreter as tflite
except ImportError:
    try:
        # Legacy TFLite Runtime
        import tflite_runtime.interpreter as tflite
    except ImportError:
        # Full TensorFlow fallback
        import tensorflow.lite as tflite

# Initialize Human body detection (using HOG for small footprint)
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

class ModelHandler:
    def __init__(self, model_path):
        print("----- FALL DETECTION MODEL INITIALIZATION -----")
        print("Model path:", model_path)

        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.input_shape = self.input_details[0]['shape']
        self.input_height = self.input_shape[1]
        self.input_width = self.input_shape[2]

        print("Model loaded successfully")
        print("Input shape:", self.input_shape)

    # -------- IMAGE PREPROCESSING --------
    def preprocess_image(self, img):
        frame_resized = cv2.resize(img, (self.input_width, self.input_height))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        input_data = np.expand_dims(frame_rgb, axis=0)

        dtype = self.input_details[0]['dtype']
        if dtype == np.float32:
            input_data = input_data.astype(np.float32) / 255.0
        else:
            input_data = input_data.astype(np.uint8)

        return input_data

    # -------- PREDICTION --------
    def predict(self, image_bytes):
        try:
            # Convert bytes → OpenCV image
            nparr = np.frombuffer(image_bytes, np.uint8)
            img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img_bgr is None:
                raise ValueError("Image decode failed")

            input_data = self.preprocess_image(img_bgr)

            # Set tensor
            self.interpreter.set_tensor(
                self.input_details[0]['index'], input_data
            )

            # Run inference
            self.interpreter.invoke()

            # Get output
            output_data = self.interpreter.get_tensor(
                self.output_details[0]['index']
            )[0]

            pred_class = int(np.argmax(output_data))
            confidence = float(output_data[pred_class])

            # -------- FALL DETECTION LOGIC --------
            CONF_THRESH = 0.80
            is_fall = bool(pred_class == 0 and confidence > CONF_THRESH)

            return {
                "is_fall": is_fall,
                "confidence": int(confidence * 100),
                "class_id": pred_class,
                "raw_confidence": confidence,
                "status": "Fall Detected" if is_fall else "Normal"
            }
        except Exception as e:
            print("Prediction error:", str(e))
            return {
                "is_fall": False,
                "confidence": 0,
                "error": str(e)
            }

# -------- INITIALIZE MODEL --------
# Try multiple possible paths to accommodate different deployment environments
paths_to_try = [
    os.path.join(os.path.dirname(__file__), "../Fall_Detection/tflite-model-maker-falldetect-model.tflite"),
    os.path.join(os.getcwd(), "Fall_Detection/tflite-model-maker-falldetect-model.tflite"),
    os.path.join(os.path.dirname(__file__), "..", "Fall_Detection", "tflite-model-maker-falldetect-model.tflite")
]

handler = None
for model_path in paths_to_try:
    if os.path.exists(model_path):
        try:
            print(f"Loading model from: {model_path}")
            handler = ModelHandler(model_path)
            break
        except Exception as e:
            print(f"Failed to load model from {model_path}: {e}")

if handler is None:
    print("CRITICAL: Model file not found in any expected location.")
