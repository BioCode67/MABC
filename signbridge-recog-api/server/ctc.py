"""연속 수어 인식 (CTC) — `ctc-v1` 모델이 오면 붙는 자리.

signbridge `ml/export_onnx.py`(task=ctc)가 내보내는 규격을 그대로 따른다:
    input  `input`  [B, T, 155] float32 — 시간축 동적
    output `output` [B, T', C]  로짓     — T' ≈ ceil(T / conv_stride)
    meta.json: task="ctc", feature_dim=155, num_classes, labels, blank_id=0, conv_stride,
               zero_depth(true면 z 채널을 0으로 넣어야 함), val_wer, trained_epoch

디코딩은 `ml/signbridge/metrics.py::ctc_greedy_decode` · `src/recognition/ctcRecognizer.ts::greedyDecode`와
같은 규칙이다 — **직전 프레임과 같으면 합치고, blank는 버리되 previous는 갱신한다.**
(결과의 마지막과 비교하면 "학교 … 학교"가 하나로 뭉개진다.)

파일이 없으면 `get_ctc()`는 None을 돌려주고 서버는 그냥 뜬다(`/health`의 `ctc_loaded`가 false).
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

import numpy as np

from .features import FEATURE_DIM, zero_depth
from .korean import gloss_label

ASSET_DIR = Path(os.environ.get("ASSET_DIR", Path(__file__).resolve().parent.parent / "assets"))
CTC_ONNX = Path(os.environ.get("CTC_ONNX", ASSET_DIR / "ctc" / "model.onnx"))
CTC_META = Path(os.environ.get("CTC_META", ASSET_DIR / "ctc" / "meta.json"))


def greedy_decode(logits: np.ndarray, blank: int) -> list[tuple[int, int, float, np.ndarray]]:
    """(T', C) 로짓 → [(클래스, 출력 프레임, 그 프레임 softmax 확신, 그 프레임 확률 벡터)]."""
    best = logits.argmax(axis=1)
    out = []
    previous = -1
    for t, token in enumerate(best.tolist()):
        if token != previous and token != blank:
            row = logits[t] - logits[t].max()
            p = np.exp(row)
            p /= p.sum()
            out.append((token, t, float(p[token]), p))
        previous = token
    return out


class CtcRecognizer:
    def __init__(self, model_path: Path = CTC_ONNX, meta_path: Path = CTC_META, threads: int | None = None):
        import onnxruntime as ort

        self.meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
        if self.meta.get("task") != "ctc":
            raise RuntimeError(f"연속 인식 모델이 아닙니다 (task={self.meta.get('task')})")
        if self.meta.get("feature_dim") != FEATURE_DIM:
            raise RuntimeError(f"특징 차원 불일치: 모델 {self.meta.get('feature_dim')} / 서버 {FEATURE_DIM}")
        self.labels: list[str] = self.meta["labels"]
        self.blank = int(self.meta.get("blank_id") or 0)
        self.stride = max(1, int(self.meta.get("conv_stride") or 1))
        self.zero_depth = bool(self.meta.get("zero_depth", False))
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads or max(1, (os.cpu_count() or 2))
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(model_path), so, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self._lock = threading.Lock()
        # 워밍업 — 브라우저(ctcRecognizer.ts)와 같이 실제 쓸 만한 길이로 한 번 돌린다.
        frames = max(int(self.meta.get("seq_len") or 32), 128)
        self.logits(np.zeros((frames, FEATURE_DIM), np.float32))

    def logits(self, feats: np.ndarray) -> np.ndarray:
        """(T, 155) → (T', C) 로짓."""
        x = np.ascontiguousarray(feats, dtype=np.float32)
        if self.zero_depth:
            x = zero_depth(x)
        with self._lock:
            out = self.session.run(None, {self.input_name: x[None]})[0]
        return out[0]

    def decode(self, feats: np.ndarray, fps: float, topk: int = 5) -> list:
        from .recognizer import Word  # 순환 import 회피

        if len(feats) < 2:
            return []
        lg = self.logits(feats)
        tokens = greedy_decode(lg, self.blank)
        steps = len(lg)
        words = []
        for n, (cls, t, conf, p) in enumerate(tokens):
            nxt = tokens[n + 1][1] if n + 1 < len(tokens) else steps
            start = t * self.stride / fps
            end = max(start + 1.0 / fps, min(nxt, steps) * self.stride / fps)
            p_alt = p.copy()
            p_alt[self.blank] = -1.0  # 후보에서 blank 제외
            idx = np.argsort(-p_alt)[:topk]
            alts = [{"gloss": self.labels[i], "label": gloss_label(self.labels[i]), "confidence": round(float(p[i]), 4)} for i in idx]
            lab = self.labels[cls]
            words.append(Word(lab, gloss_label(lab), conf, start, end, alts))
        return words

    def info(self) -> dict:
        return {
            "name": "signbridge ctc-v1",
            "num_classes": self.meta.get("num_classes"),
            "val_wer": self.meta.get("val_wer"),
            "trained_epoch": self.meta.get("trained_epoch"),
            "conv_stride": self.stride,
            "zero_depth": self.zero_depth,
        }


_instance: CtcRecognizer | None = None
_tried = False
_lock = threading.Lock()


def get_ctc() -> CtcRecognizer | None:
    """모델 파일이 있으면 1회 적재, 없으면 None. 적재 실패도 None(로그 남김)."""
    global _instance, _tried
    if _instance is not None or _tried:
        return _instance
    with _lock:
        if _instance is not None or _tried:
            return _instance
        _tried = True
        if not (CTC_ONNX.exists() and CTC_META.exists()):
            print(f"[ctc] 모델 없음 ({CTC_ONNX}) — mode=ctc 비활성")
            return None
        try:
            t = time.perf_counter()
            _instance = CtcRecognizer(CTC_ONNX, CTC_META)
            print(f"[ctc] 적재 {(time.perf_counter() - t) * 1000:.0f}ms: {_instance.info()}")
        except Exception as error:  # noqa: BLE001
            print(f"[ctc] 적재 실패: {type(error).__name__}: {error}")
        return _instance


def reset_for_tests() -> None:
    global _instance, _tried
    _instance, _tried = None, False
