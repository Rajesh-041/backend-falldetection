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

# Global state for temporal smoothing
fall_streak = 0
CONFIRMATION_THRESHOLD = 8  # Reduced slightly for faster real-time response
executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)

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
        status="Confirmed Fall",
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
    
    # Fast I/O: Read bytes immediately
    contents = await file.read()
    
    # Inference (This is the bottleneck, but ThreadPool helps if multiple frames arrive)
    prediction = handler.predict(contents)
    
    if prediction.get("is_fall"):
        fall_streak += 1
    else:
        fall_streak = max(0, fall_streak - 1) # Soft reset (decrements instead of 0) for better continuity

    is_confirmed = False
    if fall_streak >= CONFIRMATION_THRESHOLD:
        is_confirmed = True
        # Offload DB and Webhook to background so we can return '200 OK' immediately
        if fall_streak == CONFIRMATION_THRESHOLD:
            background_tasks.add_task(process_alert, prediction["confidence"])
            print("!!! FALL CONFIRMED !!!")

    return {
        "is_fall": prediction.get("is_fall", False),
        "confirmed_fall": is_confirmed,
        "current_streak": fall_streak,
        "confidence": prediction.get("confidence", 0)
    }

@app.get("/records")
def get_records():
    db = SessionLocal()
    records = db.query(FallRecord).order_by(FallRecord.timestamp.desc()).limit(10).all()
    db.close()
    return records

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
