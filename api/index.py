from fastapi import FastAPI, UploadFile, File, BackgroundTasks, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from database import SessionLocal, FallRecord
from model_handler import handler
import uvicorn
import datetime
import os
import requests

app = FastAPI()

# -------- SECURITY CONFIG --------
API_KEY = os.getenv("API_KEY", "fall-detection-secret-2026")

# -------- CORS FOR FRONTEND --------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------- BACKGROUND ALERT TASK --------
def process_alert(prediction_confidence):
    db = SessionLocal()
    new_record = FallRecord(
        status="Confirmed Fall (Pose-Verified)",
        confidence=prediction_confidence,
        timestamp=datetime.datetime.now()
    )
    db.add(new_record)
    db.commit()
    db.close()

    WEBHOOK_URL = os.getenv("WEBHOOK_URL", "")
    if WEBHOOK_URL:
        try:
            requests.post(WEBHOOK_URL, json={"status": "FALL_DETECTED", "confidence": prediction_confidence})
        except Exception as e:
            print("Webhook failed:", e)


# -------- FALL DETECTION API --------
@app.post("/detect")
async def detect_fall(
    background_tasks: BackgroundTasks, 
    file: UploadFile = File(...), 
    x_api_key: str = Header(None)
):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")

    if handler is None:
        return {"error": "Model not loaded", "is_fall": False, "confidence": 0}

    try:
        contents = await file.read()
        prediction = handler.predict(contents)

        is_fall = prediction.get("is_fall", False)
        confidence = prediction.get("confidence", 0)
        class_id = prediction.get("class_id", -1)

        # Trigger DB record if a fall is detected with reasonable confidence
        if is_fall and confidence > 70:
            background_tasks.add_task(process_alert, confidence)

        return {
            "is_fall": is_fall,
            "confidence": confidence,
            "class_id": class_id,
            "status": prediction.get("status", "Detecting"),
            "model_loaded": handler is not None
        }

    except Exception as e:
        print("Detection error:", e)
        return {"is_fall": False, "confidence": 0, "status": "error", "error": str(e)}


# -------- FETCH RECORDS --------
@app.get("/records")
def get_records(x_api_key: str = Header(None)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    db = SessionLocal()
    records = db.query(FallRecord).order_by(FallRecord.timestamp.desc()).limit(10).all()
    db.close()
    return records


# -------- ROOT ROUTE --------
@app.get("/")
def root():
    return {
        "message": "Fall Detection API Running",
        "model_loaded": handler is not None,
        "api_key_configured": API_KEY != "fall-detection-secret-2026"
    }


# -------- RUN SERVER --------
if __name__ == "__main__":
    uvicorn.run("index:app", host="0.0.0.0", port=8000, reload=True)
