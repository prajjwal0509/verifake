"""FastAPI backend for VeriFake."""

from __future__ import annotations

import os
import socket
import threading
import time
import webbrowser
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from src.image_io import (
    ALLOWED_FORMATS,
    MAX_IMAGE_PIXELS,
    MAX_IMAGE_SIDE,
    MAX_UPLOAD_BYTES,
    ImageValidationError,
    decode_image_bytes,
    register_image_plugins,
)
from src.ensemble_inference import (
    STEP3_ENSEMBLE_PATH,
    STEP4_CALIBRATION_PATH,
    EnsembleDeepfakeDetector,
)
from src.inference import DeepfakeDetector
from src.utils import (
    CLASS_NAMES_PATH,
    METRICS_PATH,
    MODELS_DIR,
    PROJECT_ROOT,
    REPORTS_DIR,
    get_logger,
    load_class_mapping,
    load_json,
)

logger = get_logger("app")


def _pick_port(host: str, preferred: int = 8000) -> int:
    """Return preferred port if free, otherwise the next available local port."""
    for p in range(preferred, preferred + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind((host, p))
                return p
            except OSError:
                continue
    return preferred


HOST = os.environ.get("DEEPFAKE_HOST", "127.0.0.1")
_ENV_PORT = os.environ.get("DEEPFAKE_PORT") or os.environ.get("PORT")
PORT = int(_ENV_PORT) if _ENV_PORT else _pick_port(HOST, 8000)
URL = f"http://localhost:{PORT}"
STATIC_DIR = PROJECT_ROOT / "app" / "static"
MAX_BYTES = MAX_UPLOAD_BYTES

register_image_plugins()

state: dict[str, Any] = {
    "detector": None,
    "detector_mode": None,
    "class_mapping": None,
    "load_error": None,
}


def _open_browser_when_ready(url: str, port: int, timeout: float = 20.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                break
        except OSError:
            time.sleep(0.25)
    chrome_mac = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    if os.path.exists(chrome_mac):
        webbrowser.register(
            "chrome-mac", None, webbrowser.BackgroundBrowser(chrome_mac)
        )
    for name in ("chrome-mac", "chrome", "google-chrome", "chromium"):
        try:
            if webbrowser.get(name).open(url, new=2):
                logger.info("Opened %s in %s", url, name)
                return
        except webbrowser.Error:
            continue
    logger.warning("Chrome not found; opening default browser")
    webbrowser.open(url, new=2)


def _ensemble_artifacts_ready() -> bool:
    return (
        STEP3_ENSEMBLE_PATH.exists()
        and STEP4_CALIBRATION_PATH.exists()
        and (MODELS_DIR / "clip_mlp_v2.pt").exists()
        and (MODELS_DIR / "convnext_v2.pt").exists()
    )


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Load class mapping and detector at startup."""
    register_image_plugins()
    try:
        state["class_mapping"] = load_class_mapping(CLASS_NAMES_PATH)
        pref = os.environ.get("VERIFAKE_DETECTOR", "auto").strip().lower()
        use_ensemble = (pref == "ensemble") or (
            pref == "auto" and _ensemble_artifacts_ready()
        )
        if use_ensemble:
            state["detector"] = await run_in_threadpool(EnsembleDeepfakeDetector)
            state["detector_mode"] = "ensemble"
        else:
            state["detector"] = await run_in_threadpool(DeepfakeDetector)
            state["detector_mode"] = "legacy"
    except (
        Exception
    ) as exc:  # Keep serving the UI with a degraded status if weights fail to load.
        state["load_error"] = str(exc)
        logger.error("Model failed to load: %s", exc)
    if os.environ.get("DEEPFAKE_NO_BROWSER") != "1":
        threading.Thread(
            target=_open_browser_when_ready, args=(URL, PORT), daemon=True
        ).start()
    yield


app = FastAPI(title="VeriFake", version="1.0.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/reports", StaticFiles(directory=REPORTS_DIR), name="reports")


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    """Serve the single-page frontend."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
async def health() -> dict[str, Any]:
    """Return liveness and detector configuration."""
    det = state["detector"]
    mode = state.get("detector_mode") or (
        "ensemble" if isinstance(det, EnsembleDeepfakeDetector) else "legacy"
    )
    band = list(getattr(det, "uncertain_band", [0.40, 0.60])) if det else [0.40, 0.60]
    thr = float(getattr(det, "decision_threshold", 0.5)) if det else None
    tta_views = 2 if isinstance(det, EnsembleDeepfakeDetector) else 3
    return {
        "status": "ok" if det else "degraded",
        "model_loaded": det is not None,
        "detector_mode": mode if det else None,
        "model": det.meta["backbone"] if det else None,
        "device": str(det.device) if det else None,
        "class_to_idx": det.class_to_idx if det else None,
        "class_names": det.class_names if det else None,
        "temperature": det.temperature if det else None,
        "decision_threshold": thr,
        "uncertain_band": band,
        "face_gate": True,
        "tta_views": tta_views,
        "max_upload_mb": MAX_BYTES // (1024 * 1024),
        "max_image_side": MAX_IMAGE_SIDE,
        "max_image_pixels": MAX_IMAGE_PIXELS,
        "supported_formats": sorted(ALLOWED_FORMATS),
        "error": state["load_error"],
    }


@app.get("/api/metrics")
async def metrics() -> JSONResponse:
    """Return test-set metrics generated by src.evaluate."""
    if not METRICS_PATH.exists():
        raise HTTPException(404, "Metrics not found; run `python -m src.evaluate`.")
    return JSONResponse(load_json(METRICS_PATH))


def _validate_and_decode(data: bytes) -> tuple[Image.Image, str]:
    try:
        return decode_image_bytes(data, max_bytes=MAX_BYTES)
    except ImageValidationError as exc:
        raise HTTPException(exc.status_code, exc.detail) from exc


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict[str, Any]:
    """Verify provenance, detect face, and run calibrated inference with Grad-CAM."""
    det = state["detector"]
    if det is None:
        raise HTTPException(503, f"Model not loaded: {state['load_error']}")
    data = await file.read(MAX_BYTES + 1)
    img, detected_format = _validate_and_decode(data)
    t0 = time.perf_counter()
    result = await run_in_threadpool(
        det.predict_with_gradcam,
        img,
        require_face=True,
        raw_bytes=data,
        use_tta=True,
        detected_format=detected_format,
    )
    result["inference_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    result["filename"] = file.filename
    result["detected_format"] = detected_format

    prov = result.get("provenance") or {}
    if result.get("status") == "no_face":
        logger.info(
            "%s (%s): no_face, provenance=%s, %.0f ms",
            file.filename,
            detected_format,
            prov.get("status"),
            result["inference_ms"],
        )
        return result

    pix = result.get("pixel_model") or {}
    logger.info(
        "%s (%s): %s (pixel=%s p_fake=%.3f, c2pa=%s, %.0f ms)",
        file.filename,
        detected_format,
        result.get("verdict_display"),
        pix.get("label"),
        (pix.get("probabilities") or {}).get("FAKE", -1.0),
        prov.get("status"),
        result["inference_ms"],
    )
    return result


def main() -> None:
    """Start uvicorn server."""
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORT, log_level="info")


if __name__ == "__main__":
    main()
