# SignBridge 수어 인식 API — Hugging Face Spaces (Docker SDK)
#
# HF Spaces 규칙: 7860 포트 · uid 1000(user) · 쓰기 가능한 곳은 /home/user와 /tmp뿐.
# 무료 CPU basic(2 vCPU · 16GB)에서 충분히 돈다 — 모델 26MB, 추론 2~7ms, MediaPipe ~30ms/프레임.
FROM python:3.11-slim

# opencv-headless·mediapipe가 요구하는 최소 시스템 라이브러리.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 libgl1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    ASSET_DIR=/home/user/app/assets \
    MAX_SECONDS=30 \
    TARGET_FPS=30

WORKDIR /home/user/app

# 의존성 먼저 — 소스가 바뀌어도 이 레이어는 캐시된다.
COPY --chown=user requirements.txt ./
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user server/ ./server/
COPY --chown=user assets/ ./assets/

EXPOSE 7860

# 자산이 없으면 URL에서 받은 뒤 기동. 워커 1개 — MediaPipe·ORT가 각자 스레드를 쓴다.
CMD ["sh", "-c", "python -m server.fetch_assets && uvicorn server.app:app --host 0.0.0.0 --port 7860 --workers 1"]
