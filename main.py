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
CONFIRMATION_THRESHOLD = 3 # Sequence Length for confirmation
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
    
    # Extract prediction data
    is_fall = prediction.get("is_fall", False)
    class_id = prediction.get("class_id", -1)
    confidence = prediction.get("confidence", 0)

    # Robust Streak Logic: Leaky Bucket
    if is_fall:
        fall_streak += 1
    else:
        # Instead of resetting to 0, just decrease it slowly
        # This prevents a single bad frame from stopping the alert
        if fall_streak > 0:
            fall_streak -= 1 

    # Cap the streak at the threshold + 1 for stability
    if fall_streak > CONFIRMATION_THRESHOLD + 1:
        fall_streak = CONFIRMATION_THRESHOLD + 1

    is_confirmed = False
    if fall_streak >= CONFIRMATION_THRESHOLD:
        is_confirmed = True
        # Only process the alert exactly when the threshold is first reached
        if fall_streak == CONFIRMATION_THRESHOLD and is_fall:
            background_tasks.add_task(process_alert, confidence)
            print(f"!!! FALL CONFIRMED !!! (Class={class_id}, Conf={confidence}%)")
    
    # Log every frame to terminal to help debug classes
    if is_fall or fall_streak > 0:
        print(f"Detecting: is_fall={is_fall} (Class={class_id}, Streak={fall_streak})")

    return {
        "is_fall": is_fall,
        "confirmed_fall": is_confirmed,
        "current_streak": fall_streak,
        "confidence": confidence,
        "status": prediction.get("status", "Detecting"),
        "class_id": class_id
    }

@app.get("/records")
def get_records():
    db = SessionLocal()
    records = db.query(FallRecord).order_by(FallRecord.timestamp.desc()).limit(10).all()
    db.close()
    return records

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
