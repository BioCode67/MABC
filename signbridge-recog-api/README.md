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
| `GET /health` | 모델·검출기 적재 상태 (`ok: true`면 준비됨) |
| `GET /labels?q=병원&limit=20` | 인식 가능한 글로스 검색 |
| `POST /recognize/video` | 영상(mp4·webm·mov) → 글로스 열 · 후보 · 문장 |
| `POST /recognize/landmarks` | 브라우저가 뽑은 랜드마크 JSON → 같은 결과 |
| `GET /docs` · `/openapi.json` | 자동 문서 / OpenAPI 스펙 |

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
검증 WER 0.22 — signbridge 저장소에 학습 완료)의 일이며, 이 서버는 그 모델이 오면 `mode=ctc`로 붙일 자리를 비워 두었습니다.

투표 디코더 수치는 같은 17클립으로 72개 설정을 훑어 고른 값이라 **낙관적**입니다(다만 thr 0.3~0.4·최소 길이 0.2~0.4s·후보 3~5개
어느 조합이든 0.64±0.01로 평탄했고, thr 0.5부터 급락). 실제 촬영본에서는 더 낮게 봐야 합니다.

### 자원 (실측, 4코어 샌드박스)

| 항목 | 값 |
|---|---|
| 기동 후 RSS | ≈ 390 MB (ONNX + MediaPipe 랜드마커 적재) |
| 요청 처리 중 정상 상태 RSS | ≈ 530 MB (랜드마커 1개 재사용, 추론 배치 32) — 요청을 거듭해도 늘지 않음 |
| MediaPipe 랜드마크 | ≈ 15 ms/프레임 (60프레임 0.9 s). 무료 Space 2 vCPU에서는 두 배쯤 잡을 것 |
| 인식(투표 디코더 + 거울 판정) | 23 s 클립 ≈ 2~4 s |
| 지원 컨테이너 | mp4(H.264) · webm(VP8/VP9) · mov · 60fps는 30fps로 솎음 — OpenCV 내장 FFmpeg, 시스템 ffmpeg 불필요 |

무료 Space(16 GB)는 넉넉합니다. **512 MB짜리 무료 PaaS(Render 무료 등)에는 맞지 않습니다** — MediaPipe만으로 넘칩니다.

## 배포 (Hugging Face Spaces, 무료 CPU)

1. **+ New Space** → SDK **Docker** → Hardware **CPU basic (free)** → Public
2. 이 폴더를 통째로 올린다 (`assets/`의 40MB 세 파일 포함 — 한도 안)
3. 첫 빌드 3~5분. `GET /health`가 `{"ok": true, ...}`면 끝

### 절전 대비 (심사 실격 방지)

무료 Space는 **48시간 미사용 시 잠들고 깨는 데 30초~1분** 걸립니다. 심사 시점 실행 오류는 실격입니다.
- 제출 후 심사 기간 동안 `GET /health`를 **주기적으로 호출**해 깨워 둡니다 (예: cron-job.org 등 무료 핑 서비스, 20분 간격)
- 발표 직전에도 한 번 열어 둡니다
- 확실히 하려면 **Persistent hardware(유료, 시간당 과금)** 로 올리면 잠들지 않습니다

### 로컬 실행

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
uvicorn server.app:app --port 7860
python tests/test_e2e.py          # signbridge 저장소가 옆에 있으면 실데이터로 정확도까지 검증
```

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
