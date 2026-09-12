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
from server.recognizer import MIN_FRAMES, PROB_BATCH, Recognizer  # noqa: E402

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

print()
if FAILS:
    print(f"FAILED {len(FAILS)}:")
    for f in FAILS:
        print("  -", f)
    sys.exit(1)
print("ALL PASS")
