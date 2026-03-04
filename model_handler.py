import os
import cv2
import numpy as np

# -------- LOAD TFLITE --------
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite


class ModelHandler:

    def __init__(self, model_path):

        print("Loading Fall Detection Model...")
        print("Model path:", model_path)

        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.input_shape = self.input_details[0]['shape']
        self.input_height = self.input_shape[1]
        self.input_width = self.input_shape[2]

        print("Model loaded successfully")

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

    def predict(self, image_bytes):

        try:
            # Convert bytes → OpenCV image
            nparr = np.frombuffer(image_bytes, np.uint8)
            img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img_bgr is None:
                raise ValueError("Image decode failed")

            input_data = self.preprocess_image(img_bgr)

            self.interpreter.set_tensor(
                self.input_details[0]['index'], input_data
            )

            self.interpreter.invoke()

            output_data = self.interpreter.get_tensor(
                self.output_details[0]['index']
            )[0]

            pred_class = np.argmax(output_data)
            confidence = float(output_data[pred_class])

            # -------- FALL DETECTION LOGIC --------
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

            print("Prediction error:", e)

            return {
                "is_fall": False,
                "confidence": 0,
                "error": str(e)
            }


# -------- INITIALIZE MODEL --------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(
    BASE_DIR,
    "Fall_Detection",
    "tflite-model-maker-falldetect-model.tflite"
)

handler = None

if os.path.exists(model_path):

    try:
        handler = ModelHandler(model_path)

    except Exception as e:

        print("Failed to initialize model:", e)

else:

    print("Model file not found:", model_path)
