import requests
import os
import json

# Configuration
BACKEND_URL = "http://localhost:8000"
TEST_IMAGE_PATH = "test_frame.jpg"

def verify_system():
    print("--- Fall Detection System Verification ---")
    
    # 1. Check if backend is reachable
    try:
        response = requests.get(f"{BACKEND_URL}/records")
        if response.status_code == 200:
            print("[SUCCESS] Backend is reachable.")
            print(f"[INFO] Current records in DB: {len(response.json())}")
        else:
            print(f"[ERROR] Backend returned status code: {response.status_code}")
            return
    except requests.exceptions.ConnectionError:
        print("[ERROR] Backend is NOT running. Please start it with 'python3 main.py' first.")
        return

    # 2. Create a dummy image for testing
    import numpy as np
    import cv2
    dummy_frame = np.zeros((224, 224, 3), dtype=np.uint8)
    cv2.putText(dummy_frame, "TEST", (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imwrite(TEST_IMAGE_PATH, dummy_frame)

    # 3. Test the Detection Endpoint
    print("[INFO] Sending test frame to /detect...")
    try:
        with open(TEST_IMAGE_PATH, 'rb') as f:
            files = {'file': (TEST_IMAGE_PATH, f, 'image/jpeg')}
            detect_res = requests.post(f"{BACKEND_URL}/detect", files=files)
            
        if detect_res.status_code == 200:
            result = detect_res.json()
            print(f"[SUCCESS] Detection Response: {json.dumps(result, indent=2)}")
            
            if "error" in result:
                print(f"[WARNING] Model Error: {result['error']}")
            elif result.get("is_fall"):
                print("[ALERT] Model detected a fall in the test frame!")
            else:
                print("[INFO] Model processed the frame correctly (No fall detected).")
        else:
            print(f"[ERROR] Detection failed with status: {detect_res.status_code}")
    except Exception as e:
        print(f"[ERROR] An error occurred during detection: {e}")
    finally:
        if os.path.exists(TEST_IMAGE_PATH):
            os.remove(TEST_IMAGE_PATH)

if __name__ == "__main__":
    verify_system()
