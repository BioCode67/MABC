"""SignBridge 수어 인식 API (FastAPI).

    GET  /health                 모델·검출기 적재 상태
    GET  /labels?q=&limit=       클래스 목록(검색)
    POST /recognize/video        영상 파일(mp4·webm·mov) → 글로스 열 + 문장
    POST /recognize/landmarks    브라우저가 뽑은 랜드마크 JSON → 글로스 열 + 문장

MABC 결선에서 Hermes Agent(Upstage Console)가 **외부 비-LLM API**로 호출한다.
LLM은 이 서버에 없다 — 문장 다듬기·대응 생성은 Solar Pro 4의 몫이고, 이 서버는
"무슨 낱말을 어떤 확신으로 읽었는지"를 정직하게 돌려준다(후보 포함).
"""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .ctc import get_ctc
from .korean import gloss_label, glosses_to_korean
from .landmarks import MAX_SECONDS_DEFAULT, TASK_PATH, extract_video, frames_from_json, get_backend
from .recognizer import META_PATH, MODEL_PATH, CtcUnavailable, get_recognizer

STATIC_DIR = Path(__file__).resolve().parent / "static"

MAX_UPLOAD_MB = float(os.environ.get("MAX_UPLOAD_MB", "80"))
ALLOWED_SUFFIX = {".mp4", ".webm", ".mov", ".avi", ".mkv", ".m4v"}

app = FastAPI(
    title="SignBridge Sign Recognition API",
    version="1.0.0",
    description="한국수어 영상/랜드마크 → 글로스 열 → 한국어 문장 (iso-v2, 13,576 클래스)",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.on_event("startup")
def _warm():
    # 첫 요청에서 수십 초를 쓰지 않도록 기동 시 미리 적재한다(심사위원은 3회만 실행한다).
    try:
        get_recognizer()
        get_backend()
        get_ctc()
        print("[app] 모델·검출기 적재 완료")
    except Exception as error:  # noqa: BLE001
        print(f"[app] 기동 시 적재 실패: {type(error).__name__}: {error}")


@app.get("/health")
def health():
    ok_model = MODEL_PATH.exists() and META_PATH.exists()
    info = {"ok": bool(ok_model), "model_path": str(MODEL_PATH), "task_path": str(TASK_PATH), "task_exists": TASK_PATH.exists()}
    try:
        r = get_recognizer()
        info.update(
            {
                "model_loaded": True,
                "num_classes": r.meta.get("num_classes"),
                "val_top1_signer_disjoint": r.meta.get("val_top1"),
                "trained_epoch": r.meta.get("trained_epoch"),
            }
        )
    except Exception as error:  # noqa: BLE001
        info.update({"model_loaded": False, "error": f"{type(error).__name__}: {error}", "ok": False})
    try:
        info["landmark_backend"] = get_backend().name
    except Exception as error:  # noqa: BLE001
        info["landmark_backend"] = f"unavailable: {type(error).__name__}"
        info["ok"] = False
    ctc = get_ctc()
    info["ctc_loaded"] = ctc is not None
    if ctc is not None:
        info["ctc"] = ctc.info()
    info["max_seconds"] = MAX_SECONDS_DEFAULT
    return info


@app.get("/labels")
def labels(q: str = "", limit: int = 50):
    r = get_recognizer()
    items = r.labels
    if q:
        items = [l for l in items if q in l]
    return {"total": len(r.labels), "matched": len(items), "labels": items[: max(1, min(limit, 500))]}


def _validate_mode(mode: str) -> str:
    mode = (mode or "segmental").lower()
    if mode not in ("segmental", "dp", "stream", "ctc"):
        raise HTTPException(400, "mode는 'segmental'(이어서 수어한 영상, 투표 디코더) · 'dp'(세그먼트 DP) · 'stream'(한 낱말씩) · 'ctc'(연속 인식 모델, 배포된 경우)")
    if mode == "ctc" and get_ctc() is None:
        raise HTTPException(409, "연속 인식(ctc) 모델이 이 서버에 배포되지 않았습니다 — mode=segmental을 쓰세요")
    return mode


def _validate_mirror(mirror: str) -> str:
    mirror = (mirror or "auto").lower()
    if mirror not in ("auto", "off", "on"):
        raise HTTPException(400, "mirror는 'auto'(기본, 거울상이면 자동 반전) · 'off' · 'on'")
    return mirror


@app.post("/recognize/video")
async def recognize_video(
    file: UploadFile = File(..., description="수어 영상 (mp4·webm·mov·avi)"),
    mode: str = Form("segmental"),
    topk: int = Form(5),
    max_seconds: float = Form(0.0, description="0이면 서버 기본(MAX_SECONDS)"),
    mirror: str = Form("auto", description="auto(기본)·off·on — 전면 카메라 거울상 영상 자동 반전"),
):
    mode = _validate_mode(mode)
    mirror = _validate_mirror(mirror)
    topk = max(1, min(int(topk), 10))
    suffix = Path(file.filename or "").suffix.lower() or ".mp4"
    if suffix not in ALLOWED_SUFFIX:
        raise HTTPException(415, f"지원하지 않는 형식 {suffix} (mp4·webm·mov·avi)")

    t0 = time.perf_counter()
    data = await file.read()
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"파일이 큽니다 (최대 {MAX_UPLOAD_MB:.0f}MB)")
    if not data:
        raise HTTPException(400, "빈 파일")

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        try:
            vl = extract_video(tmp_path, max_seconds if max_seconds > 0 else None)
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
        t1 = time.perf_counter()
        try:
            result = get_recognizer().recognize(vl.frames, vl.fps, mode=mode, topk=topk, mirror=mirror)
        except CtcUnavailable as error:
            raise HTTPException(409, str(error)) from error
        result["video"] = {
            "filename": file.filename,
            "bytes": len(data),
            "source_fps": round(vl.source_fps, 2),
            "frames_decoded": vl.frames_total,
            "frames_sampled": len(vl.frames),
            "seconds_used": round(len(vl.frames) / max(vl.fps, 1e-6), 2),
            "landmark_backend": vl.backend,
        }
        result["timing_ms"]["landmarks"] = round((t1 - t0) * 1000, 1)
        result["timing_ms"]["total"] = round((time.perf_counter() - t0) * 1000, 1)
        return result
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


class LandmarksRequest(BaseModel):
    frames: list[dict] = Field(..., description="프레임 배열: {pose:[[x,y,z]*33], leftHand:[[x,y,z]*21]|null, rightHand:..., t:ms}")
    fps: float = Field(20.0, gt=0, le=240, description="프레임 열의 실효 fps")
    mode: str = "segmental"
    topk: int = 5
    mirror: str = "auto"


@app.post("/recognize/landmarks")
def recognize_landmarks(req: LandmarksRequest):
    mode = _validate_mode(req.mode)
    mirror = _validate_mirror(req.mirror)
    if not req.frames:
        raise HTTPException(400, "frames가 비었습니다")
    if len(req.frames) > 20000:
        raise HTTPException(413, "프레임이 너무 많습니다 (최대 20,000)")
    frames = frames_from_json(req.frames)
    try:
        return get_recognizer().recognize(frames, req.fps, mode=mode, topk=max(1, min(req.topk, 10)), mirror=mirror)
    except CtcUnavailable as error:
        raise HTTPException(409, str(error)) from error


class SentenceRequest(BaseModel):
    glosses: list[str] = Field(..., description="글로스 ID 열 (예: ['머리1','어제1','아프다1'])")


@app.post("/sentence")
def sentence(req: SentenceRequest):
    """글로스 열 → 규칙 기반 문장. 사용자가 후보로 낱말을 고친 뒤 문장을 다시 만들 때 쓴다."""
    glosses = [g for g in req.glosses if isinstance(g, str) and g.strip()][:200]
    return {"glosses": glosses, "labels": [gloss_label(g) for g in glosses], "sentence": glosses_to_korean(glosses) if glosses else ""}


@app.get("/", response_class=HTMLResponse)
def root():
    """폰·PC에서 바로 찍어 올려 보는 테스트 페이지. 에이전트가 파일을 못 넘길 때의 우회 경로이기도 하다."""
    page = STATIC_DIR / "index.html"
    if page.exists():
        return HTMLResponse(page.read_text(encoding="utf-8"))
    return HTMLResponse("<p>SignBridge Sign Recognition API — see /docs</p>")


@app.get("/info")
def info():
    return {
        "service": "SignBridge Sign Recognition API",
        "endpoints": ["/", "/health", "/labels", "/sentence", "/recognize/video", "/recognize/landmarks", "/docs", "/openapi.json"],
    }
