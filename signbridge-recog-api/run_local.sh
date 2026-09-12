#!/usr/bin/env bash
# 로컬 실행 — venv 만들고 의존성 깔고 7860에서 띄운다.
#   bash run_local.sh            # 서버
#   bash run_local.sh test       # 검증 스크립트만
set -euo pipefail
cd "$(dirname "$0")"
[ -d .venv ] || python3 -m venv .venv
. .venv/bin/activate
pip install -q -r requirements.txt
if [ "${1:-}" = "test" ]; then
  exec python tests/test_e2e.py "${2:-../signbridge}"
fi
python -m server.fetch_assets
exec uvicorn server.app:app --host 0.0.0.0 --port "${PORT:-7860}"
