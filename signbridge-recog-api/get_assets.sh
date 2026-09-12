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
  # 연속 인식(CTC) 모델은 있을 때만 — 없어도 서버는 뜬다(mode=ctc만 409).
  #   만들기: signbridge에서 `bash ml/jobs/deploy_ctc.sh ~/sbruns/ctc-v1` → public/models/ksl-ctc/
  for CTC in "$SB/public/models/ksl-ctc" "$HOME/sbruns/ctc-v1/onnx-deploy"; do
    if [ -f "$CTC/model.onnx" ] && [ -f "$CTC/meta.json" ]; then
      mkdir -p assets/ctc && cp -v "$CTC/model.onnx" "$CTC/meta.json" assets/ctc/ && echo "ctc 모델 복사: $CTC" && break
    fi
  done
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
if os.path.exists("assets/ctc/meta.json"):
    c=json.load(open("assets/ctc/meta.json")); assert c.get("task")=="ctc" and c.get("feature_dim")==155, "ctc meta 규격 불일치"
    print("ctc OK · classes",c.get("num_classes"),"epoch",c.get("trained_epoch"),"val_wer",c.get("val_wer"),"zero_depth",c.get("zero_depth"))
else:
    print("ctc 모델 없음 — mode=ctc 비활성(정상)")
PY
