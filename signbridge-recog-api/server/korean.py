"""글로스 열 → 읽을 만한 한국어 문장.

BioCode67/signbridge `src/agents/glossToKorean.ts`·`glossLabel.ts`의 파이썬 포팅이다.
규칙은 셋뿐이고 일부러 적게 둔다 — **문장을 지어내지 않고 있는 낱말만 잇는다.**
  1) 글로스 번호·기호를 뗀다 (아프다1 → 아프다)
  2) 마지막 용언은 공손형으로(아프다 → 아파요), 앞의 용언은 이어지는 형태로(가다 → 가고)
  3) 나머지는 그대로 띄어 쓴다

MABC 결선에서는 Solar Pro 4가 최종 문장을 다듬을 수 있으므로, 이 함수는
LLM 없이도 읽히는 **폴백 문장**과, LLM에 넘길 **정직한 원문(글로스)**을 함께 주는 용도다.
"""

from __future__ import annotations

import re

_TAIL = re.compile(r"[0-9#:@]+$")


def gloss_label(gloss: str) -> str:
    """글로스 ID → 사람이 읽는 표제어. 숫자만으로 된 글로스(숫자 수어)는 그대로 둔다."""
    label = _TAIL.sub("", gloss)
    return label or gloss


_JUNG = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
_FUSE = {"ㅣ": "ㅕ", "ㅗ": "ㅘ", "ㅜ": "ㅝ", "ㅚ": "ㅙ"}
_BRIGHT = {"ㅏ", "ㅗ", "ㅑ", "ㅛ"}

# 불규칙은 손으로 못박는다 — 규칙대로 굴리면 "고맙아요"가 나간다. 자주 쓰는 것만.
_FIXED = {
    "아니": "아니에요", "고맙": "고마워요", "반갑": "반가워요", "춥": "추워요", "덥": "더워요",
    "맵": "매워요", "무섭": "무서워요", "아프": "아파요", "낫": "나아요", "붓": "부어요",
    "걷": "걸어요", "듣": "들어요", "묻": "물어요", "돕": "도와요", "있": "있어요",
    "없": "없어요", "괜찮": "괜찮아요", "모르": "몰라요", "부르": "불러요", "빠르": "빨라요",
}

_QUESTION = {
    "무엇": "뭐예요", "어디": "어디예요", "언제": "언제예요", "누구": "누구예요",
    "얼마": "얼마예요", "몇": "몇이에요", "왜": "왜요", "어떻게": "어떻게 해요",
}


def _decompose(ch: str):
    code = ord(ch) - 0xAC00
    if code < 0 or code >= 11172:
        return None
    return code // 588, (code % 588) // 28, code % 28


def _compose(cho: int, jung: int, jong: int) -> str:
    return chr(0xAC00 + cho * 588 + jung * 28 + jong)


def polite(stem: str) -> str:
    """용언 어간 → 공손형: 아프 → 아파요 · 가 → 가요 · 먹 → 먹어요 · 하 → 해요."""
    parts = _decompose(stem[-1]) if stem else None
    if parts is None:
        return f"{stem}요"
    cho, jung, jong = parts
    vowel = _JUNG[jung]
    if stem.endswith("하"):
        return f"{stem[:-1]}해요"
    if stem in _FIXED:
        return _FIXED[stem]
    if jong:
        return stem + ("아요" if vowel in _BRIGHT else "어요")
    if vowel == "ㅡ" and len(stem) >= 2:
        prev = _decompose(stem[-2])
        bright = prev is not None and _JUNG[prev[1]] in _BRIGHT
        return f"{stem[:-1]}{_compose(cho, _JUNG.index('ㅏ' if bright else 'ㅓ'), 0)}요"
    if vowel in ("ㅏ", "ㅓ", "ㅐ", "ㅔ"):
        return f"{stem}요"
    fused = _FUSE.get(vowel)
    if fused:
        return f"{stem[:-1]}{_compose(cho, _JUNG.index(fused), 0)}요"
    return f"{stem}요"


def _connective(stem: str) -> str:
    return f"{stem}고"


def _is_verb(lemma: str) -> bool:
    return len(lemma) >= 2 and lemma.endswith("다")


def glosses_to_korean(glosses: list[str]) -> str:
    """글로스 열 → 문장. TS `glossesToKorean`과 같은 규칙."""
    lemmas = [gloss_label(g) for g in glosses]
    lemmas = [l for l in lemmas if l]
    if not lemmas:
        return ""
    verb_flags = [_is_verb(l) for l in lemmas]
    last_verb = max((i for i, v in enumerate(verb_flags) if v), default=-1)
    words: list[str] = []
    for i, lemma in enumerate(lemmas):
        if not verb_flags[i]:
            words.append(lemma)
            continue
        stem = lemma[:-1]
        words.append(polite(stem) if i == last_verb else _connective(stem))

    if last_verb < 0:
        last = lemmas[-1]
        if last in _QUESTION:
            head = " ".join(words[:-1])
            return f"{head} {_QUESTION[last]}?" if head else f"{_QUESTION[last]}?"
        return f"{' '.join(words)}요"

    joined: list[str] = []
    for i, w in enumerate(words):
        is_hada = lemmas[i] == "하다" and joined and not verb_flags[i - 1]
        if is_hada:
            joined[-1] += w
        else:
            joined.append(w)
    return " ".join(joined)
