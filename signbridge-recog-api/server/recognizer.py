"""랜드마크 프레임 열 → 글로스 열 (ONNX 단어 인식기 + 세 가지 디코더).

모델: BioCode67/signbridge `iso-v2` (13,576 클래스, 검증 top-1 0.782, 수어자 분리).
입력 `[B, 32, 155]` float32 → 출력 로짓 `[B, 13576]`. 속도 특징은 모델 내부에서 만든다.

| 디코더 | 언제 | 방식 | 실측(AI Hub 17클립, 191낱말) |
|---|---|---|---|
| `segmental` (기본) | 이어서 수어한 영상(오프라인) | 다중 스케일 창 → 프레임별 사후확률 투표 → 런 길이 | **F1 0.648**, recall 0.73, precision 0.58 |
| `dp` | 위와 같음(예전 기본) | 다중 스케일 창 → 세그먼트 DP | F1 0.606 |
| `stream` | 한 낱말씩 끊어 수어할 때 | 브라우저 `useRecognizer.ts`와 **같은 규칙** | F1 0.584 |

세 수치 모두 **연속 수어 클립**에 단어 모델을 댄 값이다. 낱말 하나를 끊어서 하면
top-1 0.80·top-5 0.92(같은 클립의 글로스 구간 191개)가 나온다. 연속 수어를 제대로
읽는 것은 CTC 모델(`ctc-v1`, WER 0.22)의 일이며 이 서버는 그 자리를 비워 두었다.

좌우 반전(거울) 입력: 모델은 손 방향에 강하게 묶여 있다(반전하면 top-1 0.80 → 0.22).
전면 카메라 저장본이 거울상인 경우가 있어 `mirror="auto"`가 기본이다 — 원본/반전
양쪽의 창 최대확률 평균을 비교해 여유(MIRROR_MARGIN) 이상 차이 날 때만 반전을 택한다.
17클립 전부 원본을 골랐고(마진 +0.06~+0.41), 반전본을 넣으면 전부 반전을 골랐다.
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .features import FEATURE_DIM, SEQ_LEN, frame_to_features, mirror_features, resample_sequence
from .korean import gloss_label, glosses_to_korean
from .landmarks import LandmarkFrame

ASSET_DIR = Path(os.environ.get("ASSET_DIR", Path(__file__).resolve().parent.parent / "assets"))
MODEL_PATH = Path(os.environ.get("MODEL_ONNX", ASSET_DIR / "model.onnx"))
META_PATH = Path(os.environ.get("MODEL_META", ASSET_DIR / "meta.json"))

# ── stream 디코더 상수 — src/recognition/useRecognizer.ts·labels.ts와 같아야 한다 ──
BUFFER_CAP = 48
INFER_SEC = 0.25
MIN_FRAMES = 12
HAND_MIN_RATIO = 0.25
CONFIDENCE_THRESHOLD = 0.6
STABLE_HITS = 2

# ── segmental 디코더 상수 — 17클립 스윕에서 고른 값(F1 0.606) ──
SEG_SCALES_SEC = (0.4, 0.6, 0.9, 1.3, 1.8)
SEG_STEP_DIV = 4
SEG_FLOOR = 0.5  # 후보 최소 확률
SEG_AGREE = 2  # 같은 라벨의 겹치는 후보 최소 개수
SEG_SKIP_COST = 0.08  # 건너뛴 프레임당 비용 σ
SEG_WORD_PENALTY = 0.0  # 낱말당 벌점 λ

# 추론 배치 상한 — 메모리 아레나 크기를 묶는다(아래 probs 설명).
PROB_BATCH = 32

# ── vote 디코더(segmental 기본) — 72조합 스윕에서 고른 값(F1 0.648; thr 0.3~0.4·min_len 0.2~0.4·topk 3~5는 0.64±0.01로 평탄, thr 0.5부터 급락) ──
VOTE_THRESHOLD = 0.4  # 런의 평균 사후확률 하한
VOTE_MIN_LEN_SEC = 0.3  # 런 최소 길이
VOTE_TOPK = 3  # 창마다 투표에 넣는 상위 후보 수
VOTE_WEIGHT_CONF = True  # 창의 top-1 확률로 투표 가중

# ── 거울 자동 판정 ──
MIRROR_SCALE_SEC = 1.3
MIRROR_STEP_DIV = 2
MIRROR_MARGIN = 0.08  # 반전 점수가 원본 점수보다 이만큼 높아야 반전 채택(클립 마진 최소 +0.098, 낱말 하나짜리 입력 손실 ≈0.01)


@dataclass
class Word:
    gloss: str
    label: str
    confidence: float
    start: float
    end: float
    alts: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "gloss": self.gloss,
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "alts": self.alts,
        }


class Recognizer:
    def __init__(self, model_path: Path = MODEL_PATH, meta_path: Path = META_PATH, threads: int | None = None):
        import onnxruntime as ort

        self.meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
        self.labels: list[str] = self.meta["labels"]
        if self.meta.get("feature_dim") != FEATURE_DIM or self.meta.get("seq_len") != SEQ_LEN:
            raise RuntimeError(
                f"특징 규격 불일치: 모델 {self.meta.get('feature_dim')}×{self.meta.get('seq_len')} / "
                f"서버 {FEATURE_DIM}×{SEQ_LEN}"
            )
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads or max(1, (os.cpu_count() or 2))
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(model_path), so, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self._lock = threading.Lock()
        # 워밍업 — 첫 추론은 수백 ms.
        self._logits(np.zeros((1, SEQ_LEN, FEATURE_DIM), np.float32))

    # ── 기본 추론 ────────────────────────────────────────────────────────
    def _logits(self, x: np.ndarray) -> np.ndarray:
        with self._lock:
            return self.session.run(None, {self.input_name: x})[0]

    def probs(self, segments: list[np.ndarray]) -> np.ndarray:
        """(T_i, 155) 구간들 → (N, C) 확률.

        PROB_BATCH개씩 잘라 돌린다. segmental 디코더는 한 클립에서 창을 수백 개 만드는데,
        이를 한 배치로 넣으면 onnxruntime 메모리 아레나가 그 크기로 커진 채 남는다
        (실측 sign_1 686프레임: 한 배치 601MB / 64개 260MB / 32개 185MB, 속도는 32개가 더 빠름).
        """
        out = []
        for i in range(0, len(segments), PROB_BATCH):
            x = np.stack([resample_sequence(np.ascontiguousarray(s)) for s in segments[i : i + PROB_BATCH]]).astype(np.float32)
            lg = self._logits(x)
            lg = lg - lg.max(axis=1, keepdims=True)
            p = np.exp(lg)
            out.append(p / p.sum(axis=1, keepdims=True))
        return np.concatenate(out, axis=0) if out else np.zeros((0, len(self.labels)), np.float32)

    def topk(self, p: np.ndarray, k: int) -> list[dict]:
        idx = np.argsort(-p)[:k]
        return [{"gloss": self.labels[i], "label": gloss_label(self.labels[i]), "confidence": round(float(p[i]), 4)} for i in idx]

    # ── 특징 ─────────────────────────────────────────────────────────────
    @staticmethod
    def frames_to_features(frames: list[LandmarkFrame]) -> tuple[np.ndarray, np.ndarray, int]:
        """프레임 열 → (유효 특징 (T', 155), 유효 프레임의 시각 ms (T',), 무효 프레임 수).

        어깨 미검출 프레임은 브라우저처럼 **버린다**(링버퍼에 넣지 않는다).
        """
        feats, stamps = [], []
        dropped = 0
        for fr in frames:
            f = None
            if fr.pose is not None:
                # 미검출 관절이 (0,0,0)으로 들어오는 입력(OpenPose 변환·외부 JSON)에서
                # 존재 마스크 없이 정규화하면 `(0 - center) * inv` — 몸 위치에 따라 매번
                # 달라지는 값이 들어간다. 관절별 존재 여부를 좌표에서 유도해 넘긴다.
                # MediaPipe 실입력은 정확히 0이 나오지 않으므로 영향이 없다.
                pose_present = np.any(fr.pose != 0.0, axis=1)
                left = fr.left if (fr.left is not None and np.any(fr.left != 0.0)) else None
                right = fr.right if (fr.right is not None and np.any(fr.right != 0.0)) else None
                f = frame_to_features(fr.pose, left, right, pose_present)
            if f is None:
                dropped += 1
                continue
            feats.append(f)
            stamps.append(fr.t_ms)
        if not feats:
            return np.zeros((0, FEATURE_DIM), np.float32), np.zeros((0,), np.int64), dropped
        return np.stack(feats), np.asarray(stamps, dtype=np.int64), dropped

    @staticmethod
    def hand_ratio(feats: np.ndarray) -> float:
        if len(feats) == 0:
            return 0.0
        return float(np.mean((feats[:, -2] > 0) | (feats[:, -1] > 0)))

    # ── stream 디코더 (브라우저 규칙 그대로) ───────────────────────────────
    def stream_decode(self, feats: np.ndarray, fps: float, topk: int = 5) -> list[Word]:
        step = max(1, int(round(fps * INFER_SEC)))
        buf: list[np.ndarray] = []
        pending = ("", 0)
        last = ""
        out: list[Word] = []
        win_start_idx = 0
        for t in range(len(feats)):
            buf.append(feats[t])
            if len(buf) > BUFFER_CAP:
                buf.pop(0)
            if t % step or len(buf) < MIN_FRAMES:
                continue
            ratio = float(np.mean([(f[-2] > 0) or (f[-1] > 0) for f in buf]))
            if ratio < HAND_MIN_RATIO:
                pending = ("", 0)
                continue
            p = self.probs([np.stack(buf)])[0]
            i = int(p.argmax())
            c = float(p[i])
            lab = self.labels[i]
            if c < CONFIDENCE_THRESHOLD:
                pending = ("", 0)
                continue
            pending = (lab, pending[1] + 1) if pending[0] == lab else (lab, 1)
            if pending[1] >= STABLE_HITS and lab != last:
                last = lab
                win_start_idx = max(0, t - len(buf) + 1)
                out.append(Word(lab, gloss_label(lab), c, win_start_idx / fps, t / fps, self.topk(p, topk)))
        return out

    # ── segmental 디코더 (오프라인 · 다중 스케일 창 + DP) ───────────────────
    def _candidates(self, feats: np.ndarray, fps: float) -> list[tuple[int, int, int, float]]:
        T = len(feats)
        out: list[tuple[int, int, int, float]] = []
        for sc in SEG_SCALES_SEC:
            w = max(4, int(round(fps * sc)))
            step = max(1, w // SEG_STEP_DIV)
            starts = list(range(0, max(1, T - w + 1), step))
            segs = [feats[a : a + w] for a in starts]
            segs = [s for s in segs if len(s) >= 4]
            if not segs:
                continue
            P = self.probs(segs)
            for a, p in zip(starts, P):
                i = int(p.argmax())
                if p[i] >= SEG_FLOOR:
                    out.append((a, min(T, a + w), i, float(p[i])))
        return out

    @staticmethod
    def _agree(cands, min_support: int):
        if min_support <= 1:
            return cands
        keep = []
        for a, b, i, p in cands:
            n = sum(1 for a2, b2, i2, _ in cands if i2 == i and min(b, b2) - max(a, a2) > 0)
            if n >= min_support:
                keep.append((a, b, i, p))
        return keep

    @staticmethod
    def _dp(cands, T: int, lam: float, sigma: float):
        by_end: dict[int, list] = {}
        for a, b, i, p in cands:
            by_end.setdefault(b, []).append((a, i, p))
        best = [0.0] * (T + 1)
        back: list = [None] * (T + 1)
        for t in range(1, T + 1):
            best[t] = best[t - 1] - sigma
            back[t] = None
            for a, i, p in by_end.get(t, []):
                v = best[a] + math.log(p) - lam
                if v > best[t]:
                    best[t] = v
                    back[t] = (a, i, p)
        out = []
        t = T
        while t > 0:
            if back[t] is None:
                t -= 1
            else:
                a, i, p = back[t]
                out.append((a, t, i, p))
                t = a
        out.reverse()
        merged = []
        for a, b, i, p in out:
            if merged and merged[-1][2] == i:
                merged[-1] = (merged[-1][0], b, i, max(merged[-1][3], p))
            else:
                merged.append((a, b, i, p))
        return merged

    # ── vote 디코더 (오프라인 · 다중 스케일 창 → 프레임별 사후확률 투표 → 런 길이) ──
    def _posterior(self, feats: np.ndarray, fps: float) -> np.ndarray:
        """모든 창의 상위 VOTE_TOPK 확률을 창이 덮는 프레임에 더해 (T, C) 사후확률을 만든다."""
        T = len(feats)
        acc = np.zeros((T, len(self.labels)), np.float32)
        cnt = np.zeros(T, np.float32)
        for sc in SEG_SCALES_SEC:
            w = max(4, int(round(fps * sc)))
            step = max(1, w // SEG_STEP_DIV)
            starts = [a for a in range(0, max(1, T - w + 1), step) if len(feats[a : a + w]) >= 4]
            if not starts:
                continue
            P = self.probs([feats[a : a + w] for a in starts])
            for a, p in zip(starts, P):
                top = np.argpartition(-p, VOTE_TOPK)[:VOTE_TOPK]
                wgt = float(p[top].max()) if VOTE_WEIGHT_CONF else 1.0
                acc[a : a + w, top] += wgt * p[top]
                cnt[a : a + w] += wgt
        cnt[cnt == 0] = 1.0
        return acc / cnt[:, None]

    def vote_decode(self, feats: np.ndarray, fps: float, topk: int = 5) -> list[Word]:
        if len(feats) < 4:
            return []
        post = self._posterior(feats, fps)
        lab = post.argmax(axis=1)
        conf = post.max(axis=1)
        min_len = max(2, int(round(fps * VOTE_MIN_LEN_SEC)))
        runs: list[tuple[int, int, int]] = []  # (a, b, label)
        t, T = 0, len(feats)
        while t < T:
            u = t
            while u < T and lab[u] == lab[t]:
                u += 1
            if u - t >= min_len and float(conf[t:u].mean()) >= VOTE_THRESHOLD:
                if runs and runs[-1][2] == lab[t]:
                    runs[-1] = (runs[-1][0], u, lab[t])
                else:
                    runs.append((t, u, int(lab[t])))
            t = u
        words: list[Word] = []
        for a, b, i in runs:
            pm = post[a:b].mean(axis=0)
            words.append(Word(self.labels[i], gloss_label(self.labels[i]), float(pm[i]), a / fps, b / fps, self.topk(pm, topk)))
        return words

    # ── 거울 자동 판정 ───────────────────────────────────────────────────
    def orientation_score(self, feats: np.ndarray, fps: float) -> float:
        """한 스케일 창들의 최대확률 평균 — 모델이 이 방향을 얼마나 '알아보는가'."""
        w = max(4, int(round(fps * MIRROR_SCALE_SEC)))
        step = max(1, w // MIRROR_STEP_DIV)
        segs = [feats[a : a + w] for a in range(0, max(1, len(feats) - w + 1), step)]
        segs = [s for s in segs if len(s) >= 4]
        if not segs:
            return 0.0
        return float(self.probs(segs).max(axis=1).mean())

    def resolve_mirror(self, feats: np.ndarray, fps: float, mirror: str) -> tuple[np.ndarray, bool, float | None]:
        """mirror: 'off' | 'on' | 'auto' → (쓸 특징, 반전 여부, 점수 차(auto일 때))."""
        if mirror == "on":
            return mirror_features(feats), True, None
        if mirror != "auto" or len(feats) < 4:
            return feats, False, None
        mirrored = mirror_features(feats)
        margin = self.orientation_score(mirrored, fps) - self.orientation_score(feats, fps)
        if margin >= MIRROR_MARGIN:
            return mirrored, True, margin
        return feats, False, margin

    def segmental_decode(self, feats: np.ndarray, fps: float, topk: int = 5) -> list[Word]:
        if len(feats) < 4:
            return []
        cands = self._agree(self._candidates(feats, fps), SEG_AGREE)
        chosen = self._dp(cands, len(feats), SEG_WORD_PENALTY, SEG_SKIP_COST)
        if not chosen:
            return []
        # 고른 구간마다 top-k 후보를 다시 뽑는다(사용자가 고쳐 쓸 수 있게 — 브라우저와 같은 UX).
        P = self.probs([feats[a:b] for a, b, _, _ in chosen])
        words: list[Word] = []
        for (a, b, i, _), p in zip(chosen, P):
            lab = self.labels[i]
            words.append(Word(lab, gloss_label(lab), float(p[i]), a / fps, b / fps, self.topk(p, topk)))
        return words

    # ── 한 번에 ────────────────────────────────────────────────────────────
    def decode(self, feats: np.ndarray, fps: float, mode: str, topk: int) -> list[Word]:
        if mode == "stream":
            return self.stream_decode(feats, fps, topk)
        if mode == "dp":
            return self.segmental_decode(feats, fps, topk)
        return self.vote_decode(feats, fps, topk)  # "segmental"

    def recognize(self, frames: list[LandmarkFrame], fps: float, mode: str = "segmental", topk: int = 5, mirror: str = "auto") -> dict:
        t0 = time.perf_counter()
        feats, _stamps, dropped = self.frames_to_features(frames)
        ratio = self.hand_ratio(feats)
        status = "ok"
        words: list[Word] = []
        mirrored, margin = False, None
        if len(feats) < MIN_FRAMES:
            status = "no_pose"  # 어깨가 잡힌 프레임이 너무 적다 — 상반신이 화면에 없다
        elif ratio < HAND_MIN_RATIO:
            status = "no_hands"  # 손이 안 보이면 답을 내지 않는다(브라우저와 같은 원칙)
        else:
            feats, mirrored, margin = self.resolve_mirror(feats, fps, mirror)
            words = self.decode(feats, fps, mode, topk)
        glosses = [w.gloss for w in words]
        return {
            "status": status,
            "mode": mode,
            "mirror": mirror,
            "mirrored": mirrored,
            "mirror_margin": None if margin is None else round(margin, 4),
            "words": [w.to_dict() for w in words],
            "glosses": glosses,
            "labels": [w.label for w in words],
            "sentence": glosses_to_korean(glosses) if glosses else "",
            "frames_input": len(frames),
            "frames_valid": int(len(feats)),
            "frames_dropped_no_pose": int(dropped),
            "hand_ratio": round(ratio, 3),
            "fps": round(float(fps), 2),
            "timing_ms": {"recognize": round((time.perf_counter() - t0) * 1000, 1)},
            "model": {
                "name": "signbridge iso-v2",
                "num_classes": self.meta.get("num_classes"),
                "val_top1_signer_disjoint": self.meta.get("val_top1"),
                "trained_epoch": self.meta.get("trained_epoch"),
            },
        }


_instance: Recognizer | None = None
_instance_lock = threading.Lock()


def get_recognizer() -> Recognizer:
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = Recognizer()
    return _instance
