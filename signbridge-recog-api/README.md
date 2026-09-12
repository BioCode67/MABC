---
title: SignBridge Sign Recognition API
emoji: 🤟
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: apache-2.0
short_description: 한국수어 영상 → 글로스 열 → 한국어 문장 (iso-v2, 13,576 클래스)
---

# SignBridge Sign Recognition API

한국수어(KSL) **영상 또는 MediaPipe 랜드마크**를 받아 **글로스 열 + 후보 + 읽을 만한 한국어 문장**을 돌려주는
비-LLM API입니다. MABC 2026 결선에서 Hermes Agent가 외부 도구로 호출합니다.

```
수어 영상 ─▶ MediaPipe Holistic(.task, 브라우저와 동일) ─▶ 155차원 특징 ─▶ iso-v2 ONNX ─▶ 글로스 열
                                                                             └▶ 규칙 기반 문장(폴백)
```

- 모델: [BioCode67/signbridge](https://github.com/BioCode67/signbridge) `iso-v2` — AI Hub 재난안전+수어영상 217만 표본, **13,576 클래스**,
  검증(수어자 분리) **top-1 0.782**. 학습 코드·특징 정의(`features.py`)를 그대로 가져왔습니다.
- LLM 없음. 문장 다듬기·대화 대응은 호출 측(Solar Pro 4)이 합니다. 이 서버는 **무엇을 얼마나 확신하며 읽었는지**를 돌려줍니다.

## 엔드포인트

| 경로 | 하는 일 |
|---|---|
| `GET /` | **테스트 페이지** — 폰·PC에서 바로 촬영/업로드 → 결과 확인 → 후보로 낱말 수정 → 에이전트에 붙여넣을 텍스트 복사 |
| `GET /health` | 모델·검출기 적재 상태 (`ok: true`면 준비됨) |
| `GET /labels?q=병원&limit=20` | 인식 가능한 글로스 검색 |
| `POST /recognize/video` | 영상(mp4·webm·mov) → 글로스 열 · 후보 · 문장 |
| `POST /recognize/landmarks` | 브라우저가 뽑은 랜드마크 JSON → 같은 결과 |
| `POST /sentence` | 글로스 열 → 규칙 기반 문장 (후보로 고친 뒤 다시 만들 때) |
| `GET /docs` · `/openapi.json` · `/info` | 자동 문서 / OpenAPI 스펙 / 엔드포인트 목록 |

### 테스트 페이지 (`/`)

Space 주소를 폰에서 열면 바로 씁니다. **📷 촬영·영상 선택**은 폰 카메라 앱으로 찍어 올리고(가장 확실), **브라우저에서 녹화**는
`MediaRecorder`로 8초까지 찍습니다(webm 또는 mp4). 결과의 노란 낱말은 확신이 0.5 미만이라 후보에서 골라 고칠 수 있고,
고치면 `/sentence`로 문장을 다시 만듭니다. **🔊 읽어주기**는 브라우저 내장 음성 합성(Web Speech API, `ko-KR`)으로 문장을 청인에게
들려줍니다 — 서버·외부 API 없이 폰·PC 대부분에서 됩니다. 맨 아래 텍스트는 그대로 복사해 에이전트 대화창에 붙이는 용도입니다 —
**에이전트가 영상 파일을 API로 못 넘기는 경우의 우회 경로**이자 심사위원이 즉시 써 볼 수 있는 화면입니다.

```
[수어 인식 결과]
낱말: 머리 / 어제 / 아프다
문장 초안: 머리 어제 아파요
확신 낮은 낱말: 어제(42%) → 후보: 오늘, 내일, 그제
```

### 영상 인식

```bash
curl -X POST https://<space>.hf.space/recognize/video \
  -F "file=@clip.mp4" -F "mode=segmental" -F "topk=5"
```

| 필드 | 값 | 뜻 |
|---|---|---|
| `mode` | `segmental` (기본) | **이어서 수어한 영상**. 다중 스케일 창의 확률을 프레임마다 투표해 낱말 런을 찾는다 |
|  | `dp` | 같은 용도의 예전 디코더(창 후보 + 세그먼트 DP). 비교용 |
|  | `stream` | **한 낱말씩 끊어서** 수어한 영상. 브라우저 앱과 같은 규칙(0.6 이상 2회 연속 확정) |
| `mirror` | `auto` (기본) | 원본/좌우반전 중 모델이 더 잘 알아보는 쪽을 고른다. **전면 카메라가 거울상으로 저장한 영상** 대비 |
|  | `on` / `off` | 강제 반전(왼손잡이 수어자) / 반전 안 함 |
| `topk` | 1~10 | 낱말마다 함께 돌려줄 후보 수 |
| `max_seconds` | 0=서버 기본(30) | 앞에서부터 이 길이만 본다 |

### 응답

```json
{
  "status": "ok",                       // ok · no_hands(손이 안 보임) · no_pose(상반신이 안 잡힘)
  "mode": "segmental",
  "mirror": "auto", "mirrored": false, "mirror_margin": -0.31,   // 거울상으로 판정해 반전했으면 mirrored: true
  "words": [
    {"gloss": "머리1", "label": "머리", "confidence": 0.91, "start": 0.4, "end": 1.1,
     "alts": [{"gloss": "머리1", "label": "머리", "confidence": 0.91},
              {"gloss": "허리1", "label": "허리", "confidence": 0.05}]},
    {"gloss": "아프다1", "label": "아프다", "confidence": 0.84, "start": 1.2, "end": 2.0, "alts": [...]}
  ],
  "glosses": ["머리1", "아프다1"],
  "labels":  ["머리", "아프다"],
  "sentence": "머리 아파요",             // 규칙 기반 폴백 — 지어내지 않고 있는 낱말만 잇는다
  "hand_ratio": 0.83, "frames_valid": 58, "fps": 30.0,
  "timing_ms": {"landmarks": 1840.2, "recognize": 210.5, "total": 2091.3},
  "model": {"name": "signbridge iso-v2", "num_classes": 13576, "val_top1_signer_disjoint": 0.7818}
}
```

`status`가 `no_hands`/`no_pose`면 `words`는 비어 있습니다. **손이 안 보이면 답을 내지 않습니다** — 손 없는 특징 벡터는
매 프레임 거의 같아져서 모델이 엉뚱한 낱말을 자신 있게 내기 때문입니다(브라우저 앱과 같은 원칙).

### 랜드마크 인식 (브라우저에서 MediaPipe를 이미 돌린 경우)

```json
POST /recognize/landmarks
{"fps": 20, "mode": "segmental", "topk": 5,
 "frames": [{"pose": [[x,y,z], ...33], "leftHand": [[x,y,z], ...21], "rightHand": null, "t": 0}, ...]}
```

## 실측 수치 — 정직하게

AI Hub 재난 수어 클립 17개(191 낱말 구간)로 잰 값입니다.

| 조건 | 결과 |
|---|---|
| 글로스 구간을 **정답 타임코드로 잘라** 낱말 하나씩 | **top-1 0.801 · top-5 0.916** |
| 이어서 수어한 전체 클립, `mode=stream` (브라우저 규칙) | F1 0.584 (recall 0.56 · precision 0.62) |
| 이어서 수어한 전체 클립, `mode=dp` | F1 0.606 (recall 0.70 · precision 0.54) |
| 이어서 수어한 전체 클립, `mode=segmental` (투표) | **F1 0.648 (recall 0.73 · precision 0.58)** |
| 같은 낱말 구간을 **좌우반전**해서 넣으면 | top-1 0.215 — 모델이 손 방향에 묶여 있음 |
| `mirror=auto` 판정 | 17클립 원본은 전부 원본으로, 반전본은 전부 반전으로 판정(점수 차 +0.10~+0.44) |

즉 **낱말 단위 인식은 검증 수치대로 나오고, 연속 수어를 낱말로 쪼개는 것이 병목**입니다. 그것은 CTC 연속 인식 모델(`ctc-v1`,
검증 WER 0.22 — signbridge 저장소에 학습 완료)의 일이며, 이 서버는 그 모델을 **`mode=ctc`로 바로 붙일 수 있게** 해 두었습니다.

### `mode=ctc` — 연속 인식 모델 붙이기

1. signbridge에서 `bash ml/jobs/deploy_ctc.sh ~/sbruns/ctc-v1` → `public/models/ksl-ctc/{model.onnx,meta.json}`
2. 그 두 파일을 이 폴더의 `assets/ctc/`에 둔다 (`get_assets.sh <signbridge>`가 있으면 자동으로 복사)
3. 재기동 → `GET /health`의 `ctc_loaded: true`, `mode=ctc`로 호출

규격은 `ml/export_onnx.py`가 내보내는 그대로입니다(`input [1,T,155]` → `output [1,T',C]`, `meta.json`의 `blank_id`·`conv_stride`·`zero_depth`).
디코딩은 `ml/signbridge/metrics.py`·브라우저 `ctcRecognizer.ts`와 같은 그리디 축약입니다. 모델이 없으면 `mode=ctc`는 409를 돌려주고
나머지는 그대로 돕니다. 배관은 합성 ONNX로 검증했습니다(`tests/test_e2e.py` [7]) — **실제 모델을 넣은 뒤 17클립으로 F1을 꼭 다시 재세요.**

투표 디코더 수치는 같은 17클립으로 72개 설정을 훑어 고른 값이라 **낙관적**입니다(다만 thr 0.3~0.4·최소 길이 0.2~0.4s·후보 3~5개
어느 조합이든 0.64±0.01로 평탄했고, thr 0.5부터 급락). 실제 촬영본에서는 더 낮게 봐야 합니다.

### 자원 (실측, 4코어 샌드박스)

| 항목 | 값 |
|---|---|
| 기동 후 RSS | ≈ 390 MB (ONNX + MediaPipe 랜드마커 적재) |
| 요청 처리 중 정상 상태 RSS | ≈ 530 MB (랜드마커 1개 재사용, 추론 배치 32) — 요청을 거듭해도 늘지 않음 |
| MediaPipe 랜드마크 | 사람이 있는 프레임 ≈ 50 ms (640×480), 1080p 원본은 긴 변 960으로 줄여 같은 속도(`MAX_SIDE`). 30fps 영상은 15fps로 솎는다(`TARGET_FPS`, 정확도 손실 F1 −0.01). 무료 Space 2 vCPU에서 10초 클립 ≈ 10~15초 |
| 인식(투표 디코더 + 거울 판정) | 23 s 클립 ≈ 2~4 s |
| 지원 컨테이너 | mp4(H.264) · webm(VP8/VP9) · mov — OpenCV 내장 FFmpeg, 시스템 ffmpeg 불필요 |

무료 Space(16 GB)는 넉넉합니다. **512 MB짜리 무료 PaaS(Render 무료 등)에는 맞지 않습니다** — MediaPipe만으로 넘칩니다.

### 알아 둘 것 — mediapipe 0.10.21 파이썬 래퍼의 치명적 버그

`HolisticLandmarker.detect_for_video()`는 얼굴이 잡히고 **손(또는 포즈) 패킷이 비면 프로세스가 통째로 죽습니다**
(`Check failed: holder_ != nullptr The packet is empty` — 파이썬 예외가 아니라 abort). 수어 영상은 한 손만 보이는 프레임이 흔해서
이 경로로는 첫 실제 영상에서 서버가 죽습니다(실제 사람 사진으로 재현, `tests/test_e2e.py` [6]). 얼굴이 안 잡히면 포즈·손까지 빈 결과를
돌려주는 문제도 있습니다. 그래서 `server/landmarks.py`는 래퍼를 거치지 않고 그래프를 직접 돌린 뒤 스트림마다 `is_empty()`를 확인합니다.
**mediapipe 버전을 올리거나 바꾸면 이 테스트를 꼭 다시 돌리세요.**

## 배포 (Hugging Face Spaces, 무료 CPU)

1. **+ New Space** → SDK **Docker** → Hardware **CPU basic (free)** → Public
2. 이 폴더를 통째로 올린다 (`assets/`의 40MB 세 파일 포함 — 한도 안)
   - 웹 UI "Files → Upload"로 올리면 큰 파일은 자동으로 LFS 처리된다
   - `git push`로 올리면 **10MB 넘는 파일은 LFS 필수**: `git lfs install && git lfs track "*.onnx" "*.task"` 한 뒤 add·commit
   - 자산을 안 올리고 `MODEL_ONNX_URL`·`MODEL_META_URL`·`HOLISTIC_TASK_URL` 변수(Settings → Variables)로 받게 할 수도 있다
3. 첫 빌드 3~5분. `GET /health`가 `{"ok": true, ...}`면 끝

### 절전 대비 (심사 실격 방지)

무료 Space는 **48시간 미사용 시 잠들고 깨는 데 30초~1분** 걸립니다. 심사 시점 실행 오류는 실격입니다.
- 이 저장소의 GitHub Actions `.github/workflows/keepalive.yml`이 **20분마다 `/health`를 호출**합니다.
  저장소 Settings → Secrets and variables → Actions → **Variables**에 `SIGNBRIDGE_API_URL = https://<space>.hf.space`를
  넣으면 켜집니다(없으면 조용히 건너뜀). Actions 탭에서 `keep-alive`를 수동 실행(Run workflow)해 초록불을 확인하세요.
- 발표 직전에도 한 번 열어 둡니다
- 확실히 하려면 **Persistent hardware(유료, 시간당 과금)** 로 올리면 잠들지 않습니다

### 로컬 실행

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
uvicorn server.app:app --port 7860
python tests/test_e2e.py          # signbridge 저장소가 옆에 있으면 실데이터로 정확도까지 검증
pip install onnx                  # (선택) [7] CTC 배관 검사는 합성 ONNX를 만들어야 해서 onnx 패키지가 필요
```

`requirements.txt`는 검증한 버전으로 **정확히 고정**되어 있습니다. 새 가상환경에서 설치 → 전체 테스트 통과까지 확인한 조합이니
심사 직전에 버전을 올리지 마세요.

## Hermes Agent / 다른 에이전트에 붙이기

`hermes/openapi.json`(OpenAPI 3.1)을 도구로 등록하면 됩니다. 에이전트에게 알려 줄 사용 규칙:

- 영상은 `POST /recognize/video`에 `multipart/form-data`(`file`)로 보낸다
- **`status`가 `ok`가 아니면** 인식 결과가 없는 것이다 — 사용자에게 손·상반신이 보이게 다시 찍어 달라고 안내한다
- `sentence`는 폴백 문장이다. **최종 문장은 `glosses`+`alts`를 근거로 LLM이 만들되, 없는 낱말을 지어내지 않는다**
- `confidence`가 낮은 낱말(< 0.5)은 `alts`를 사용자에게 보여 고르게 한다 — 틀린 낱말이 그대로 전달되면 안 된다
- `mirrored`가 true면 "영상이 거울상이라 반전해서 읽었다"는 뜻이다. 결과가 이상하면 `mirror=off`로 한 번 더 부를 수 있다

## 규정 관련

- 이 서버에는 LLM이 없습니다(Solar Pro 4 외 모델 호출 금지 규정과 무관). MediaPipe·ONNX 단어 분류기만 있습니다.
- 외부 API Key가 필요 없습니다(공개 Space). 제출 폼의 API Key 엑셀은 해당 없음.
- 모델·특징·규칙 코드는 참가자 본인 저장소(BioCode67/signbridge)에서 가져왔고, 데이터 출처는 AI Hub입니다.
