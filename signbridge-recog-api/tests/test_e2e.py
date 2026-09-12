"""서버 패키지 전 구간 검증 — 파이썬만으로 돈다(pytest 불필요).

    python tests/test_e2e.py [signbridge 저장소 경로]

검사 항목
  1. korean.py 규칙 7건(원본 TS 회귀 사례와 같은 것)
  2. 모델 로드·I/O 계약·추론 시간
  3. (signbridge 저장소가 있으면) AI Hub 키포인트 17클립으로
     - 글로스 구간 낱말 인식 top-1 ≥ 0.75, top-5 ≥ 0.88
     - 연속 클립 segmental(투표) F1 ≥ 0.62, dp F1 ≥ 0.55, stream 디코더 동작
     - 거울 자동 판정: 원본 17클립은 원본으로, 반전본은 반전으로
  4. /recognize/landmarks 입력 형식 왕복(JSON 프레임 → 인식)
  5. 손 없음/포즈 없음 가드
  6. 실제 사람 사진으로 만든 영상 → MediaPipe 검출(래퍼 빈 패킷 크래시 회귀) → 인식
  7. CTC 자리: 축약 규칙 · 합성 ONNX(동적 시간축·stride·zero_depth) · mode=ctc 왕복 · 모델 없을 때 409용 예외
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from server.korean import glosses_to_korean, gloss_label, polite  # noqa: E402
from server.features import mirror_features  # noqa: E402
from server.landmarks import LandmarkFrame, frames_from_json  # noqa: E402
from server.recognizer import MIN_FRAMES, PROB_BATCH, CtcUnavailable, Recognizer  # noqa: E402

FAILS: list[str] = []


def check(cond: bool, msg: str):
    print(("  ✓ " if cond else "  ✗ ") + msg)
    if not cond:
        FAILS.append(msg)


def lcs(a, b):
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            dp[i + 1][j + 1] = dp[i][j] + 1 if x == y else max(dp[i][j + 1], dp[i + 1][j])
    return dp[-1][-1]


print("[1] korean.py")
check(glosses_to_korean(["머리1", "어제1", "아프다1"]) == "머리 어제 아파요", "머리 어제 아프다 → 머리 어제 아파요")
check(glosses_to_korean(["화장실1", "어디1", "있다1"]) == "화장실 어디 있어요", "화장실 어디 있다 → 화장실 어디 있어요")
check(glosses_to_korean(["카드1", "계산1", "하다1"]) == "카드 계산해요", "카드 계산 하다 → 카드 계산해요")
check(glosses_to_korean(["이름1", "무엇1"]) == "이름 뭐예요?", "이름 무엇 → 이름 뭐예요?")
check(glosses_to_korean(["병원1", "가다1", "약1", "먹다1"]) == "병원 가고 약 먹어요", "가다·먹다 연결")
check(polite("고맙") == "고마워요" and polite("춥") == "추워요", "불규칙 고정표")
# TS `/[0-9#:@]+$/`와 동일 — **꼬리**만 뗀다. "시:9시"는 끝이 '시'라 그대로다(원본과 같은 동작).
check(gloss_label("시:9시") == "시:9시" and gloss_label("병원1@") == "병원" and gloss_label("조심1") == "조심" and gloss_label("5") == "5", "gloss_label 꼬리 제거 (TS 동일)")

print("[2] model")
t = time.perf_counter()
rec = Recognizer()
check(rec.meta["num_classes"] == len(rec.labels) == 13576, f"13,576 클래스 (로드 {(time.perf_counter()-t)*1000:.0f}ms)")
p = rec.probs([np.zeros((20, 155), np.float32)])
check(p.shape == (1, 13576) and abs(float(p.sum()) - 1.0) < 1e-4, "확률 형상·정규화")
t = time.perf_counter()
for _ in range(10):
    rec.probs([np.random.rand(32, 155).astype(np.float32)])
check((time.perf_counter() - t) / 10 < 0.05, f"추론 {(time.perf_counter()-t)/10*1000:.1f} ms/run (< 50ms)")
pb = rec.probs([np.random.rand(20, 155).astype(np.float32) for _ in range(PROB_BATCH * 2 + 3)])
check(pb.shape == (PROB_BATCH * 2 + 3, 13576) and rec.probs([]).shape == (0, 13576), f"배치 {PROB_BATCH}개씩 잘라 추론(경계·빈 입력)")

print("[3] AI Hub 실데이터 (signbridge 저장소)")
sb = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/home/user/biocode67/signbridge")
if not (sb / "public/data/sign_1.json").exists():
    print("  - 건너뜀: 저장소 없음", sb)
else:
    from server.features import clip_to_features
    from server.openpose import convert_clip

    tot = t1 = t5 = 0
    G = P = L = Ls = Ps = Pv = Lv = 0
    mir_ok = mir_n = 0
    for n in range(1, 18):
        f = sb / f"public/data/sign_{n}.json"
        if not f.exists():
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        fps = float(d["fps"])
        arr = convert_clip(d["keypoints"])
        feats, valid = clip_to_features(arr["pose"], arr["left"], arr["right"], arr["left_present"], arr["right_present"], arr["pose_present"])
        T = len(feats)
        idx = {l: i for i, l in enumerate(rec.labels)}
        for g in d["gloss_sequence"]:
            if g["gloss"] not in idx:
                continue
            a = max(0, int(round(g["start"] * fps)))
            b = min(T, int(round(g["end"] * fps)))
            if b - a < 4:
                continue
            seg = feats[a:b]
            seg = seg[valid[a:b]] if valid[a:b].any() else seg
            pp = rec.probs([seg])[0]
            top = np.argsort(-pp)[:5]
            tot += 1
            t1 += int(top[0] == idx[g["gloss"]])
            t5 += int(idx[g["gloss"]] in top)
        fv = feats[valid]
        gt = [g["gloss"] for g in d["gloss_sequence"] if g["gloss"] in idx]
        seg_words = [w.gloss for w in rec.segmental_decode(fv, fps)]
        st_words = [w.gloss for w in rec.stream_decode(fv, fps)]
        vt_words = [w.gloss for w in rec.vote_decode(fv, fps)]
        G += len(gt); P += len(seg_words); L += lcs(gt, seg_words); Ps += len(st_words); Ls += lcs(gt, st_words)
        Pv += len(vt_words); Lv += lcs(gt, vt_words)
        # 거울 자동 판정: 원본은 원본으로, 반전본은 반전으로
        _, m0, g0 = rec.resolve_mirror(fv, fps, "auto")
        _, m1, g1 = rec.resolve_mirror(mirror_features(fv), fps, "auto")
        mir_n += 2; mir_ok += int(not m0) + int(m1)
    top1, top5 = t1 / tot, t5 / tot
    f1 = 2 * L / (G + P) if G + P else 0
    f1s = 2 * Ls / (G + Ps) if G + Ps else 0
    f1v = 2 * Lv / (G + Pv) if G + Pv else 0
    check(top1 >= 0.75, f"낱말 구간 top-1 {top1:.3f} ≥ 0.75 (n={tot})")
    check(top5 >= 0.88, f"낱말 구간 top-5 {top5:.3f} ≥ 0.88")
    check(f1v >= 0.62, f"연속 클립 segmental(투표) F1 {f1v:.3f} ≥ 0.62 (recall {Lv/G:.3f} · precision {Lv/max(Pv,1):.3f})")
    check(f1 >= 0.55, f"연속 클립 dp F1 {f1:.3f} ≥ 0.55 (recall {L/G:.3f} · precision {L/max(P,1):.3f})")
    check(Ps > 0, f"연속 클립 stream F1 {f1s:.3f} (동작 확인)")
    check(mir_ok == mir_n, f"거울 자동 판정 {mir_ok}/{mir_n} (원본→원본, 반전본→반전)")

    print("[4] /recognize/landmarks 형식 왕복")
    d = json.loads((sb / "public/data/sign_1.json").read_text(encoding="utf-8"))
    arr = convert_clip(d["keypoints"])
    payload = []
    for k in range(len(arr["pose"])):
        payload.append({
            "pose": arr["pose"][k].tolist(),
            "leftHand": arr["left"][k].tolist() if arr["left_present"][k] else None,
            "rightHand": arr["right"][k].tolist() if arr["right_present"][k] else None,
            "t": int(k * 1000 / d["fps"]),
        })
    frames = frames_from_json(payload)
    res = rec.recognize(frames, float(d["fps"]), mode="segmental", topk=5)
    check(res["status"] == "ok" and len(res["words"]) >= 8, f"status={res['status']} words={len(res['words'])} sentence='{res['sentence'][:40]}…'")
    check(all(len(w["alts"]) == 5 for w in res["words"]), "낱말마다 후보 5개")
    check(all(w["end"] > w["start"] for w in res["words"]), "구간 시각 단조")
    check(res["mirror"] == "auto" and res["mirrored"] is False and res["mirror_margin"] < 0, f"기본 mirror=auto, 원본 판정(margin {res['mirror_margin']})")
    # 좌우반전한 프레임(x→1−x, 대칭 관절·양손 맞바꿈 = 거울 영상을 MediaPipe에 돌린 결과)을 넣으면 반전 판정 + 같은 낱말 열
    SWAP = [(1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16), (17, 18), (19, 20), (21, 22), (23, 24), (25, 26), (27, 28), (29, 30), (31, 32)]
    perm = list(range(33))
    for a, b in SWAP:
        perm[a], perm[b] = b, a

    def flip_pts(pts):
        return None if pts is None else [[(1 - x) if (x or y or z) else 0.0, y, z] for x, y, z in pts]

    flipped = [{"pose": flip_pts([f["pose"][perm[i]] for i in range(33)]), "leftHand": flip_pts(f["rightHand"]), "rightHand": flip_pts(f["leftHand"]), "t": f["t"]} for f in payload]
    res_m = rec.recognize(frames_from_json(flipped), float(d["fps"]), mode="segmental", topk=5)
    same = lcs(res["glosses"], res_m["glosses"]) / max(len(res["glosses"]), 1)
    check(res_m["mirrored"] is True and same >= 0.8, f"좌우반전 입력 → mirrored={res_m['mirrored']} (margin {res_m['mirror_margin']}), 낱말 일치 {same:.2f}")
    res_off = rec.recognize(frames_from_json(flipped), float(d["fps"]), mode="segmental", topk=5, mirror="off")
    check(len(res_off["glosses"]) < len(res["glosses"]) or lcs(res["glosses"], res_off["glosses"]) < lcs(res["glosses"], res_m["glosses"]), "mirror=off로 반전 입력을 읽으면 더 나쁘다(자동 판정의 효과)")
    for md in ("dp", "stream"):
        r2 = rec.recognize(frames, float(d["fps"]), mode=md, topk=3)
        check(r2["status"] == "ok" and r2["mode"] == md and all(len(w["alts"]) == 3 for w in r2["words"]), f"mode={md} words={len(r2['words'])}")

print("[5] 가드")
none_frames = [LandmarkFrame(pose=None, left=None, right=None, t_ms=i * 33) for i in range(40)]
r = rec.recognize(none_frames, 30.0)
check(r["status"] == "no_pose" and r["words"] == [], "포즈 없음 → no_pose")
pose = np.random.rand(33, 3).astype(np.float32)
pose[11] = [0.4, 0.5, 0]; pose[12] = [0.6, 0.5, 0]
hand_less = [LandmarkFrame(pose=pose, left=None, right=None, t_ms=i * 33) for i in range(40)]
r = rec.recognize(hand_less, 30.0)
check(r["status"] == "no_hands" and r["words"] == [], "손 없음 → no_hands")
few = [LandmarkFrame(pose=pose, left=np.random.rand(21, 3).astype(np.float32), right=None, t_ms=i * 33) for i in range(MIN_FRAMES - 1)]
r = rec.recognize(few, 30.0)
check(r["status"] == "no_pose", "프레임 부족 → no_pose")

print("[6] MediaPipe 실사진 경로 (0.10.21 래퍼 빈 패킷 크래시 회귀)")
img_path = ROOT / "tests" / "data" / "basketball1.png"
if not img_path.exists():
    print("  - 건너뜀: 사진 없음", img_path)
else:
    import tempfile

    import cv2

    from server.landmarks import extract_video

    img = cv2.imread(str(img_path))
    H, W = img.shape[:2]
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "real.mp4"
        vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (640, 480))
        for i in range(50):  # 2초, 천천히 줌·팬 — 정지 사진이라도 VIDEO 모드 추적을 태운다
            z = 1.0 + 0.15 * i / 50
            cx, cy = W / 2 + 40 * np.sin(i / 12), H / 2
            M = cv2.getRotationMatrix2D((cx, cy), 0, z)
            M[:, 2] += (320 - cx, 240 - cy)
            vw.write(cv2.warpAffine(img, M, (640, 480)))
        vw.release()
        t = time.perf_counter()
        vl = extract_video(out)
        dt = time.perf_counter() - t
        vl2 = extract_video(out)  # 두 번째 요청: 재사용 인스턴스 + 타임스탬프 오프셋 경로
    pose = sum(f.pose is not None for f in vl.frames)
    hands = sum((f.left is not None) or (f.right is not None) for f in vl.frames)
    missing = sum((f.left is None) or (f.right is None) for f in vl.frames)
    check(vl.backend == "mediapipe-tasks", f"백엔드 {vl.backend}")
    n = len(vl.frames)
    check(n >= 20 and pose >= 0.8 * n, f"포즈 검출 {pose}/{n} 프레임 ({dt:.1f}s, 실효 {vl.fps:.1f}fps)")
    check(hands >= 0.4 * n, f"손 검출 {hands} 프레임")
    check(missing >= 1, f"한쪽 손이 빈 프레임 {missing}개 — 빈 패킷 경로를 지나고도 살아 있음")
    r = rec.recognize(vl.frames, vl.fps)
    check(r["status"] in ("ok", "no_hands"), f"recognize status={r['status']} hand_ratio={r['hand_ratio']}")
    check(len(vl2.frames) == n and sum(f.pose is not None for f in vl2.frames) >= 0.8 * n, "두 번째 요청도 정상(랜드마커 재사용)")

print("[7] CTC 연속 인식 자리 (합성 모델로 배관 검증)")
import os  # noqa: E402
import tempfile  # noqa: E402

from server import ctc as ctc_mod  # noqa: E402
from server.ctc import greedy_decode  # noqa: E402

# 축약 규칙: 직전 프레임과 비교, blank는 버리되 previous 갱신 (ml/signbridge/metrics.py와 동일)
def _lg(seq, C=4):
    a = np.full((len(seq), C), -5.0, np.float32)
    for t, c in enumerate(seq):
        a[t, c] = 5.0
    return a
check([t[0] for t in greedy_decode(_lg([1, 1, 0, 1, 2, 2, 0, 0, 2]), 0)] == [1, 1, 2, 2], "CTC 축약: 반복 합침 · blank 사이 같은 낱말은 둘")
check([t[0] for t in greedy_decode(_lg([0, 0, 0]), 0)] == [], "CTC 축약: 전부 blank → 빈 열")
check(greedy_decode(_lg([3, 3]), 0)[0][1] == 0 and 0.99 < greedy_decode(_lg([3, 3]), 0)[0][2] <= 1.0, "CTC 토큰 프레임·확신")

try:
    import onnx
    from onnx import TensorProto, helper, numpy_helper
except ImportError:
    onnx = None
if onnx is None:
    print("  - 건너뜀: onnx 패키지 없음 (pip install onnx)")
else:
    rng = np.random.default_rng(0)
    C, H, K, STRIDE = 6, 16, 3, 2
    W1 = numpy_helper.from_array(rng.normal(0, 0.05, (H, 155, K)).astype(np.float32), "W1")
    B1 = numpy_helper.from_array(np.zeros(H, np.float32), "B1")
    W2 = numpy_helper.from_array(rng.normal(0, 0.5, (H, C)).astype(np.float32), "W2")
    B2 = numpy_helper.from_array(np.array([2.0, 0, 0, 0, 0, 0], np.float32), "B2")  # blank가 자주 이기게
    nodes = [
        helper.make_node("Transpose", ["input"], ["x_t"], perm=[0, 2, 1]),
        helper.make_node("Conv", ["x_t", "W1", "B1"], ["c"], kernel_shape=[K], strides=[STRIDE], pads=[1, 1]),
        helper.make_node("Relu", ["c"], ["r"]),
        helper.make_node("Transpose", ["r"], ["r_t"], perm=[0, 2, 1]),
        helper.make_node("MatMul", ["r_t", "W2"], ["m"]),
        helper.make_node("Add", ["m", "B2"], ["output"]),
    ]
    graph = helper.make_graph(nodes, "fake_ctc", [helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch", "frames", 155])],
                              [helper.make_tensor_value_info("output", TensorProto.FLOAT, ["batch", "frames_out", C])], initializer=[W1, B1, W2, B2])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    with tempfile.TemporaryDirectory() as td:
        mp_, me_ = Path(td) / "model.onnx", Path(td) / "meta.json"
        onnx.save(model, str(mp_))
        me_.write_text(json.dumps({"task": "ctc", "feature_dim": 155, "seq_len": 32, "zero_depth": True, "num_classes": C,
                                   "labels": ["<blank>", "병원1", "가다1", "약1", "먹다1", "머리1"], "blank_id": 0, "conv_stride": STRIDE,
                                   "trained_epoch": 1, "val_wer": 0.99}), encoding="utf-8")
        ctc_mod.reset_for_tests()
        check(ctc_mod.get_ctc() is None, "모델 파일 없으면 get_ctc() → None (기본 경로)")
        crec = ctc_mod.CtcRecognizer(mp_, me_)
        T = 97
        lg = crec.logits(np.random.rand(T, 155).astype(np.float32))
        check(lg.shape == ((T + 1) // STRIDE, C), f"동적 시간축 {T} → {lg.shape} (stride {STRIDE})")
        feats_r = np.random.rand(120, 155).astype(np.float32) * 2 - 1
        words_c = crec.decode(feats_r, 30.0, topk=3)
        check(all(w.end > w.start for w in words_c) and all(len(w.alts) == 3 and all(a["gloss"] != "<blank>" for a in w.alts) for w in words_c),
              f"ctc decode → {len(words_c)}개 낱말, 구간 단조·후보에 blank 없음")
        # zero_depth: z 채널을 0으로 넣는지 — z만 다른 두 입력의 로짓이 같아야 한다
        f2 = feats_r.copy(); f2[:, 2:153:3] += 3.0
        check(np.allclose(crec.logits(feats_r), crec.logits(f2), atol=1e-5), "meta.zero_depth=true → z 채널 무시")
        # recognize(mode='ctc') 왕복: 환경변수로 합성 모델을 가리킨다
        ctc_mod.CTC_ONNX, ctc_mod.CTC_META = mp_, me_
        ctc_mod.reset_for_tests()
        check(ctc_mod.get_ctc() is not None, "파일 있으면 get_ctc() 적재")
        pose7 = np.random.rand(33, 3).astype(np.float32)
        pose7[11] = [0.4, 0.5, 0]; pose7[12] = [0.6, 0.5, 0]
        frames7 = [LandmarkFrame(pose=pose7, left=np.random.rand(21, 3).astype(np.float32), right=None, t_ms=i * 33) for i in range(90)]
        r7 = rec.recognize(frames7, 30.0, mode="ctc", topk=3)
        check(r7["status"] == "ok" and r7["mode"] == "ctc" and r7["model"]["name"] == "signbridge ctc-v1", f"recognize(mode=ctc) status={r7['status']} words={len(r7['words'])}")
        ctc_mod.reset_for_tests()
        ctc_mod.CTC_ONNX, ctc_mod.CTC_META = Path(td) / "none.onnx", Path(td) / "none.json"
        try:
            rec.recognize(frames7, 30.0, mode="ctc")
            check(False, "모델 없을 때 mode=ctc → 예외가 나야 한다")
        except CtcUnavailable:
            check(True, "모델 없을 때 mode=ctc → CtcUnavailable")

print()
if FAILS:
    print(f"FAILED {len(FAILS)}:")
    for f in FAILS:
        print("  -", f)
    sys.exit(1)
print("ALL PASS")
