"""영상 → 프레임별 랜드마크 (MediaPipe Holistic).

브라우저(`src/recognition/holistic.ts`)와 **같은 `.task` 모델**을 MediaPipe Tasks API로 돌린다.
학습·브라우저·서버가 같은 검출기를 써야 특징 분포가 어긋나지 않는다 — 이 프로젝트에서
가장 찾기 어려운 실패("검증 정확도는 높은데 실제로는 0%")를 막는 첫 번째 방어선이다.

파이썬 Tasks API의 결과는 JS와 달리 **평탄한 리스트**다:
    result.pose_landmarks      -> List[NormalizedLandmark] (33) 또는 []
    result.left_hand_landmarks -> List[NormalizedLandmark] (21) 또는 []
(JS는 `poseLandmarks[0]`처럼 한 겹 더 감싸져 있다.)

Tasks API 초기화가 실패하면 레거시 `mp.solutions.holistic`으로 폴백한다
(`ml/etl/extract_mediapipe.py`가 쓰던 경로). 결과 형식은 같게 맞춘다.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

NUM_POSE = 33
NUM_HAND = 21

ASSET_DIR = Path(os.environ.get("ASSET_DIR", Path(__file__).resolve().parent.parent / "assets"))
TASK_PATH = Path(os.environ.get("HOLISTIC_TASK", ASSET_DIR / "holistic_landmarker.task"))

# 긴 영상은 여기서 자른다 — 심사위원이 올릴 클립은 수십 초, 서버 CPU는 2코어다.
MAX_SECONDS_DEFAULT = float(os.environ.get("MAX_SECONDS", "30"))
# 원본이 60fps처럼 높으면 30fps 근처로 내려 받는다(브라우저 실효 fps 15~25와 비슷하게).
TARGET_FPS = float(os.environ.get("TARGET_FPS", "30"))


@dataclass
class LandmarkFrame:
    pose: np.ndarray | None  # (33, 3) or None
    left: np.ndarray | None  # (21, 3) or None
    right: np.ndarray | None  # (21, 3) or None
    t_ms: int


@dataclass
class VideoLandmarks:
    frames: list[LandmarkFrame]
    fps: float  # 실효 fps (샘플링 후)
    source_fps: float
    frames_total: int  # 디코드한 원본 프레임 수(샘플링 전)
    backend: str


def _to_array(landmarks, n: int) -> np.ndarray | None:
    if not landmarks or len(landmarks) != n:
        return None
    return np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)


class _TasksHolistic:
    """MediaPipe Tasks HolisticLandmarker — 브라우저와 동일 모델.

    인스턴스는 **프로세스당 하나**를 재사용한다. 요청마다 새로 만들면 `close()`를 불러도
    메모리가 돌아오지 않는다(실측 60프레임 요청 12회: 340→866MB, 계속 증가 / 재사용: 359MB 고정,
    속도도 25% 빠름). VIDEO 모드는 타임스탬프가 단조 증가해야 하므로 요청마다 오프셋을 더하고,
    요청 사이에 검은 프레임 하나를 넣어 추적 상태(이전 영상의 관심영역)를 끊는다.
    """

    name = "mediapipe-tasks"

    def __init__(self, task_path: Path):
        import mediapipe as mp
        from mediapipe.tasks import python as mpp
        from mediapipe.tasks.python import vision

        self._mp = mp
        opts = vision.HolisticLandmarkerOptions(
            base_options=mpp.BaseOptions(model_asset_path=str(task_path)),
            running_mode=vision.RunningMode.VIDEO,
        )
        self._make = lambda: vision.HolisticLandmarker.create_from_options(opts)
        self._lm = self._make()
        self._lock = threading.Lock()
        self._next_ts = 0
        self._blank = np.zeros((64, 64, 3), dtype=np.uint8)

    def _detect(self, rgb: np.ndarray, ts: int):
        img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        return self._lm.detect_for_video(img, int(ts))

    def run_video(self, frames_rgb, timestamps_ms) -> list[LandmarkFrame]:
        out: list[LandmarkFrame] = []
        with self._lock:
            base = self._next_ts
            try:
                # 구분용 검은 프레임 — 이전 요청의 추적 상태를 지운다(검출 없음 → 다음 프레임은 재검출).
                self._detect(self._blank, base)
                base += 40
                last = 0
                for rgb, ts in zip(frames_rgb, timestamps_ms):
                    last = max(last, int(ts))
                    r = self._detect(rgb, base + int(ts))
                    out.append(
                        LandmarkFrame(
                            pose=_to_array(r.pose_landmarks, NUM_POSE),
                            left=_to_array(r.left_hand_landmarks, NUM_HAND),
                            right=_to_array(r.right_hand_landmarks, NUM_HAND),
                            t_ms=int(ts),
                        )
                    )
                self._next_ts = base + last + 40
            except Exception:
                # 랜드마커가 깨진 채 남지 않게 새로 만든다(타임스탬프도 0부터).
                try:
                    self._lm.close()
                except Exception:  # noqa: BLE001
                    pass
                self._lm = self._make()
                self._next_ts = 0
                raise
        return out


class _LegacyHolistic:
    """`mp.solutions.holistic` 폴백 — ETL(`extract_mediapipe.py`)과 같은 설정."""

    name = "mediapipe-solutions"

    def __init__(self):
        import mediapipe as mp

        self._holistic_cls = mp.solutions.holistic.Holistic

    def run_video(self, frames_rgb, timestamps_ms) -> list[LandmarkFrame]:
        out: list[LandmarkFrame] = []
        with self._holistic_cls(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            refine_face_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as h:
            for rgb, ts in zip(frames_rgb, timestamps_ms):
                r = h.process(rgb)
                pose = _to_array(r.pose_landmarks.landmark, NUM_POSE) if r.pose_landmarks else None
                left = _to_array(r.left_hand_landmarks.landmark, NUM_HAND) if r.left_hand_landmarks else None
                right = _to_array(r.right_hand_landmarks.landmark, NUM_HAND) if r.right_hand_landmarks else None
                out.append(LandmarkFrame(pose=pose, left=left, right=right, t_ms=int(ts)))
        return out


_backend = None
_backend_lock = threading.Lock()


def get_backend():
    """검출 백엔드를 1회 초기화한다. Tasks → 실패 시 레거시."""
    global _backend
    if _backend is not None:
        return _backend
    with _backend_lock:
        if _backend is not None:
            return _backend
        try:
            if not TASK_PATH.exists():
                raise FileNotFoundError(f"holistic .task 없음: {TASK_PATH}")
            _backend = _TasksHolistic(TASK_PATH)
        except Exception as error:  # noqa: BLE001 — 폴백 사유를 로그로 남긴다
            print(f"[landmarks] Tasks API 초기화 실패({type(error).__name__}: {error}) → 레거시 폴백")
            _backend = _LegacyHolistic()
        return _backend


def decode_video(path: str | Path, max_seconds: float | None = None) -> tuple[list[np.ndarray], list[int], float, float, int]:
    """영상 파일 → (RGB 프레임 목록, 타임스탬프 ms, 실효 fps, 원본 fps, 원본 프레임 수).

    OpenCV 내장 FFMPEG로 mp4/webm/mov를 읽는다(시스템 ffmpeg 불필요).
    원본 fps가 TARGET_FPS보다 높으면 정수 배로 솎아 낸다.
    """
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError("영상을 열 수 없습니다 (지원 형식: mp4·webm·mov·avi)")
    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if not (1.0 <= src_fps <= 240.0):
        src_fps = 30.0
    every = max(1, int(round(src_fps / TARGET_FPS))) if src_fps > TARGET_FPS * 1.5 else 1
    eff_fps = src_fps / every
    limit = max_seconds if max_seconds is not None else MAX_SECONDS_DEFAULT
    max_src_frames = int(limit * src_fps) if limit and limit > 0 else 0

    frames: list[np.ndarray] = []
    stamps: list[int] = []
    index = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if max_src_frames and index >= max_src_frames:
            break
        if index % every == 0:
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            stamps.append(int(round(index * 1000.0 / src_fps)))
        index += 1
    cap.release()
    if not frames:
        raise ValueError("영상에서 프레임을 읽지 못했습니다")
    return frames, stamps, eff_fps, src_fps, index


def extract_video(path: str | Path, max_seconds: float | None = None) -> VideoLandmarks:
    frames, stamps, eff_fps, src_fps, total = decode_video(path, max_seconds)
    backend = get_backend()
    lms = backend.run_video(frames, stamps)
    return VideoLandmarks(frames=lms, fps=eff_fps, source_fps=src_fps, frames_total=total, backend=backend.name)


def frames_from_json(payload: list[dict]) -> list[LandmarkFrame]:
    """브라우저가 이미 MediaPipe를 돌린 경우 — 랜드마크 JSON을 그대로 받는다.

    항목 형식(프레임당): {"pose": [[x,y,z],...33], "leftHand": [[x,y,z],...21] | null,
                         "rightHand": [...] | null, "t": ms(선택)}
    `src/recognition/landmarks.ts`의 LandmarkFrame과 같은 뜻이다.
    """
    out: list[LandmarkFrame] = []
    for k, fr in enumerate(payload):
        def arr(key, n):
            v = fr.get(key)
            if not v:
                return None
            a = np.asarray(v, dtype=np.float32)
            if a.ndim == 2 and a.shape[0] == n and a.shape[1] >= 3:
                return a[:, :3]
            if a.ndim == 1 and a.size >= n * 3:
                return a[: n * 3].reshape(n, 3)
            return None

        out.append(
            LandmarkFrame(
                pose=arr("pose", NUM_POSE),
                left=arr("leftHand", NUM_HAND) if "leftHand" in fr else arr("left", NUM_HAND),
                right=arr("rightHand", NUM_HAND) if "rightHand" in fr else arr("right", NUM_HAND),
                t_ms=int(fr.get("t", k * 33)),
            )
        )
    return out
