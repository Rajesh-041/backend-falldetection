# Fall Detection Backend

A lightweight FastAPI backend that detects falls from a single image frame using an on-device TFLite (LiteRT) model. It exposes a simple REST API that a client (e.g. a camera app or IoT device) can call with an image, and logs confirmed falls to a database for later review.

## How it works

1. A client sends an image frame to `POST /detect`.
2. The image is decoded with OpenCV and resized/normalized to match the model's expected input.
3. A `.tflite` model (loaded via `ai-edge-litert`, falling back to `tflite-runtime` or full TensorFlow) runs inference and returns a predicted class and confidence score.
4. If a fall is detected with confidence above the threshold, the event is logged asynchronously to the database via a FastAPI background task.
5. Logged events can be retrieved via `GET /records`.

All API routes (except `/`) are protected by a simple `x-api-key` header check.

## Tech stack

- **FastAPI** + **Uvicorn** — web framework / ASGI server
- **ai-edge-litert** (TFLite/LiteRT) — model inference, with `tflite_runtime` and `tensorflow.lite` as fallbacks
- **OpenCV** (`opencv-python-headless`) + **Pillow** — image decoding/preprocessing
- **SQLAlchemy** — ORM for storing fall records (SQLite by default, or any SQLAlchemy-supported DB via `DATABASE_URL`)
- **Pydantic** — used by FastAPI for request/response handling

## Project structure

```
.
├── api/
│   ├── index.py          # FastAPI app + routes (Vercel serverless entrypoint)
│   ├── model_handler.py   # Loads the TFLite model and runs inference
│   └── database.py        # SQLAlchemy engine/session and FallRecord model
├── Fall_Detection/
│   └── tflite-model-maker-falldetect-model.tflite   # Trained fall-detection model
├── model_handler.py        # Local/dev copy of the model handler
├── database.py             # Local/dev copy of the database module
├── verify_backend.py        # Script to smoke-test a running backend
├── requirements.txt
├── pyproject.toml
├── vercel.json              # Vercel routing/CORS config (entry: api/index.py)
├── .github/workflows/        # GitHub Actions workflows for Azure Web App deployment
└── package.json
```

> Note: `model_handler.py` and `database.py` exist both at the repo root and inside `api/`. The copies inside `api/` are the ones actually used when the app runs (whether via Vercel or `uvicorn`), since `api/index.py` imports them as local modules.

## API endpoints

| Method | Path       | Auth required | Description |
|--------|------------|----------------|--------------|
| GET    | `/`        | No             | Health check; reports whether the model loaded successfully. |
| POST   | `/detect`  | Yes (`x-api-key`) | Accepts an image file (`multipart/form-data`, field `file`) and returns a fall prediction. |
| GET    | `/records` | Yes (`x-api-key`) | Returns the 10 most recent logged fall records. |

### `POST /detect` response

```json
{
  "is_fall": true,
  "confidence": 92,
  "class_id": 0,
  "status": "Fall Detected",
  "model_loaded": true
}
```

## Getting started

### Prerequisites

- Python 3.12 (see `.python-version`)

### Installation

```bash
git clone https://github.com/Rajesh-041/backend-falldetection.git
cd backend-falldetection
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Configuration

Set these environment variables as needed (sensible defaults are used otherwise):

| Variable       | Default                              | Description |
|----------------|---------------------------------------|--------------|
| `API_KEY`      | `fall-detection-secret-2026`          | Required value for the `x-api-key` header on protected routes. **Change this in any real deployment.** |
| `DATABASE_URL` | `sqlite:////tmp/fall_records.db`      | SQLAlchemy database URL. Use a Postgres URL or similar for persistent, non-serverless deployments. |

### Running locally

```bash
cd api
uvicorn index:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`.

### Testing the running backend

With the server running, in another terminal:

```bash
python verify_backend.py
```

This sends a generated test frame to `/detect` and checks that the API responds correctly.

### Example request

```bash
curl -X POST http://localhost:8000/detect \
  -H "x-api-key: fall-detection-secret-2026" \
  -F "file=@frame.jpg"
```

## Deployment

This repo is set up for two deployment targets:

- **Vercel** — `vercel.json` rewrites all routes to `api/index.py` and sets permissive CORS headers, so the app deploys as a Vercel Python serverless function out of the box. Note that the default SQLite path (`/tmp/fall_records.db`) is ephemeral on serverless platforms; use an external `DATABASE_URL` for persistence.
- **Azure Web App** — `.github/workflows/` contains GitHub Actions workflows (`main_backendapi-04.yml`, `main_backendapi1.yml`) that build the app and deploy it to Azure Web App via `azure/webapps-deploy`. These require the corresponding Azure service principal secrets to be configured in the repository.

## Model

The fall-detection model lives at `Fall_Detection/tflite-model-maker-falldetect-model.tflite` (~4 MB), trained via TFLite Model Maker. `model_handler.py` loads it, resizes incoming frames to the model's expected input shape, and classifies each frame; a prediction is treated as "Fall Detected" when the predicted class is `0` and confidence exceeds the configured threshold.

## License

No license file is currently included in this repository.
