from fastapi import FastAPI, UploadFile, File, BackgroundTasks, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from database import SessionLocal, FallRecord
from model_handler import handler
import uvicorn
import datetime
import os
import requests

app = FastAPI()

# -------- SECURITY CONFIG --------
API_KEY = os.getenv("API_KEY", "fall-detection-secret-2026")

# -------- MANUAL CORS & OPTIONS HANDLER --------
@app.middleware("http")
async def add_cors_header(request: Request, call_next):
    if request.method == "OPTIONS":
        response = JSONResponse(content="OK")
    else:
        response = await call_next(request)
    
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS, PUT, DELETE"
    response.headers["Access-Control-Allow-Headers"] = "x-api-key, Content-Type, Accept"
    return response

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
        "model_loaded": handler is not None
    }

if __name__ == "__main__":
    uvicorn.run("index:app", host="0.0.0.0", port=8000, reload=True)
