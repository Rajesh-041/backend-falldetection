from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from database import SessionLocal, FallRecord
from model_handler import handler
import uvicorn
import datetime
import os
import requests
import concurrent.futures

app = FastAPI()

# Configuration
CONFIRMATION_THRESHOLD = 16 # Sequence Length for confirmation
fall_streak = 0

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def process_alert(prediction_confidence):
    """Heavy tasks handled outside the request-response cycle"""
    db = SessionLocal()
    new_record = FallRecord(
        status="Confirmed Fall (Pose-Verified)",
        confidence=prediction_confidence,
        timestamp=datetime.datetime.now()
    )
    db.add(new_record)
    db.commit()
    db.close()
    
    # Trigger Webhook
    WEBHOOK_URL = os.getenv("WEBHOOK_URL", "")
    if WEBHOOK_URL:
        try:
            requests.post(WEBHOOK_URL, json={"status": "FALL_DETECTED", "conf": prediction_confidence})
        except: pass

@app.post("/detect")
async def detect_fall(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    global fall_streak
    if handler is None: return {"error": "Model not loaded"}
    
    contents = await file.read()
    prediction = handler.predict(contents)
    
    # Logic: Only confirm if 16 consecutive frames detect a fall (Sliding Window)
    if prediction["is_fall"]:
        fall_streak += 1
    else:
        # Debounce: Instead of 0, we can use a "cooldown" or just 0
        fall_streak = 0 

    is_confirmed = False
    if fall_streak >= CONFIRMATION_THRESHOLD:
        is_confirmed = True
        if fall_streak == CONFIRMATION_THRESHOLD:
            background_tasks.add_task(process_alert, prediction["confidence"])
            print("!!! FALL CONFIRMED BY POSE & TFLITE !!!")

    return {
        "is_fall": prediction.get("is_fall", False),
        "confirmed_fall": is_confirmed,
        "current_streak": fall_streak,
        "confidence": prediction.get("confidence", 0),
        "status": prediction.get("status", "Unknown")
    }

@app.get("/records")
def get_records():
    db = SessionLocal()
    records = db.query(FallRecord).order_by(FallRecord.timestamp.desc()).limit(10).all()
    db.close()
    return records

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
