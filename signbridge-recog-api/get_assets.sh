#!/usr/bin/env bash
# 모델·검출기 자산 3개를 assets/에 채운다. 둘 중 하나:
#   bash get_assets.sh /path/to/signbridge      # 로컬 signbridge 저장소에서 복사(권장)
#   bash get_assets.sh                          # GitHub Pages 배포본에서 내려받기
set -euo pipefail
cd "$(dirname "$0")"; mkdir -p assets
if [ -n "${1:-}" ]; then
  SB="$1"
  cp -v "$SB/public/models/ksl-iso/model.onnx" "$SB/public/models/ksl-iso/meta.json" assets/
  cp -v "$SB/public/mediapipe/holistic_landmarker.task" assets/
else
  B=https://biocode67.github.io/signbridge
  curl -fL -o assets/model.onnx "$B/models/ksl-iso/model.onnx"
  curl -fL -o assets/meta.json  "$B/models/ksl-iso/meta.json"
  curl -fL -o assets/holistic_landmarker.task "$B/mediapipe/holistic_landmarker.task"
fi
ls -la assets/
python3 - <<'PY'
import json,os
m=json.load(open("assets/meta.json")); assert m["num_classes"]==13576 and m["feature_dim"]==155 and m["seq_len"]==32
assert os.path.getsize("assets/model.onnx")>20_000_000 and os.path.getsize("assets/holistic_landmarker.task")>10_000_000
print("assets OK · iso-v2 epoch",m["trained_epoch"],"val_top1",round(m["val_top1"],4))
PY
