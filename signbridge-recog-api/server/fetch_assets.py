"""배포 환경에서 모델·검출기 파일이 없으면 URL에서 받아 둔다.

기본 배치는 저장소 `assets/`에 세 파일을 **함께 커밋**하는 것이다(합계 40MB, HF 한도 안).
그래도 URL 방식을 남겨 두는 이유: 모델을 갈아 끼울 때 이미지 재빌드 없이 변수만 바꾸면 된다.

    MODEL_ONNX_URL   → $MODEL_ONNX  (기본 assets/model.onnx)
    MODEL_META_URL   → $MODEL_META  (기본 assets/meta.json)
    HOLISTIC_TASK_URL→ $HOLISTIC_TASK (기본 assets/holistic_landmarker.task)

이미 있으면 건너뛴다. 실패해도 서버는 뜬다(/health가 false로 알린다).
"""

from __future__ import annotations

import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ASSET_DIR = Path(os.environ.get("ASSET_DIR", ROOT / "assets"))

PAIRS = [
    ("MODEL_ONNX_URL", "MODEL_ONNX", ASSET_DIR / "model.onnx"),
    ("MODEL_META_URL", "MODEL_META", ASSET_DIR / "meta.json"),
    ("HOLISTIC_TASK_URL", "HOLISTIC_TASK", ASSET_DIR / "holistic_landmarker.task"),
]


def fetch(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[assets] 이미 있음: {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[assets] 받는 중: {url} → {dest}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as src, open(tmp, "wb") as out:  # noqa: S310 — 운영자가 넣은 URL
        while chunk := src.read(1 << 20):
            out.write(chunk)
    tmp.replace(dest)
    print(f"[assets] 완료: {dest} ({dest.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    for url_key, path_key, default in PAIRS:
        dest = Path(os.environ.get(path_key, default))
        url = os.environ.get(url_key)
        if dest.exists() and dest.stat().st_size > 0:
            print(f"[assets] OK: {dest}")
            continue
        if not url:
            print(f"[assets] 없음: {dest} ({url_key} 미설정) — /health가 false를 낼 것", file=sys.stderr)
            continue
        try:
            fetch(url, dest)
        except Exception as error:  # noqa: BLE001 — 하나가 실패해도 서버는 떠야 한다
            print(f"[assets] {url_key} 실패: {type(error).__name__}: {error}", file=sys.stderr)


if __name__ == "__main__":
    main()
