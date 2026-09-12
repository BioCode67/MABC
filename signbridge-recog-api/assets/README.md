# assets/ — 여기에 세 파일이 있어야 서버가 뜬다

| 파일 | 크기 | 출처 (signbridge 저장소) |
|---|---|---|
| `model.onnx` | 26 MB | `public/models/ksl-iso/model.onnx` (iso-v2, 13,576 클래스) |
| `meta.json` | 226 KB | `public/models/ksl-iso/meta.json` |
| `holistic_landmarker.task` | 13.7 MB | `public/mediapipe/holistic_landmarker.task` (브라우저와 동일 MediaPipe 모델) |

채우기: `bash get_assets.sh /path/to/signbridge` (로컬 복사) 또는 `bash get_assets.sh` (GitHub Pages에서 다운로드).
HF Space에는 이 세 파일을 **함께 올린다**(합계 40MB, 한도 안). 없으면 `/health`가 `ok:false`.
