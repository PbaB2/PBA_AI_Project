from __future__ import annotations

from typing import Any
import re

REQUIRED_SLOTS = [
    "party_purpose",
    "current_mood",
    "preferred_tastes",
    "disliked_tastes",
    "preferred_aromas",
    "disliked_aromas",
    "strength_preference",
    "favorite_drinks",
    "disliked_bases",
    "finish_preference",
]

NEGATIVE_CUES = ["싫", "안 좋아", "별로", "부담", "빼고", "제외", "피하고"]
POSITIVE_CUES = ["좋아", "좋고", "좋겠", "원해", "선호", "괜찮"]

AROMA_KEYWORDS = {
    "민트향": "민트향",
    "민트": "민트향",
    "과일향": "과일향",
    "과일": "과일향",
    "우디향": "우디향",
    "우디": "우디향",
    "커피향": "커피향",
    "커피": "커피향",
    "시트러스향": "시트러스향",
    "시트러스": "시트러스향",
    "허브향": "허브향",
    "허브": "허브향",
}

TASTE_KEYWORDS = {
    "상큼": "상큼함",
    "달콤": "단맛",
    "단맛": "단맛",
    "청량": "청량함",
    "고소": "고소함",
    "스파이시": "스파이시",
    "매콤": "스파이시",
    "밀키": "밀키함",
    "부드러운": "밀키함",
    "쓴맛": "쓴맛",
    "쌉쌀": "쓴맛",
    "신맛": "신맛",
}

BASE_KEYWORDS = {
    "위스키": "위스키",
    "럼": "럼",
    "진": "진",
    "데킬라": "데킬라",
    "보드카": "보드카",
}


def split_clauses(text: str) -> list[str]:
    parts = re.split(r"[.!?]|,| 그리고 | 근데 | 하지만 | 다만 | 그런데 ", text)
    return [p.strip() for p in parts if p.strip()]


def is_negative_clause(clause: str) -> bool:
    return any(cue in clause for cue in NEGATIVE_CUES)


def collect_tags(clause: str, keyword_map: dict[str, str]) -> list[str]:
    found = []
    for keyword, tag in keyword_map.items():
        if keyword in clause and tag not in found:
            found.append(tag)
    return found

def _is_filled(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str) and value.strip() == "":
        return False
    if isinstance(value, list) and len(value) == 0:
        return False
    if isinstance(value, dict) and len(value) == 0:
        return False
    return True


def get_completion(slots: dict) -> float:
    filled = 0
    for key in REQUIRED_SLOTS:
        if _is_filled(slots.get(key)):
            filled += 1
    return round((filled / len(REQUIRED_SLOTS)) * 100, 1)


def _merge_list_slot(old_value: Any, new_items: list[str]) -> list[str]:
    current = old_value if isinstance(old_value, list) else []
    merged = list(current)

    for item in new_items:
        if item not in merged:
            merged.append(item)

    return merged


def extract_slots_from_korean_text(user_msg: str, current_slots: dict) -> dict:
    text = user_msg.strip()
    extracted = {}

    # 1. 파티 목적
    if "생일" in text:
        extracted["party_purpose"] = "생일파티"
    elif "축하" in text:
        extracted["party_purpose"] = "축하자리"
    elif "데이트" in text:
        extracted["party_purpose"] = "데이트"
    elif "혼자" in text:
        extracted["party_purpose"] = "혼술"
    elif "친구" in text and "파티" in text:
        extracted["party_purpose"] = "친구모임"

    # 2. 현재 기분
    if "기분 좋아" in text or "신나" in text or "행복" in text:
        extracted["current_mood"] = "신남"
    elif "우울" in text or "처져" in text:
        extracted["current_mood"] = "우울함"
    elif "차분" in text or "조용" in text:
        extracted["current_mood"] = "차분함"
    elif "스트레스" in text:
        extracted["current_mood"] = "스트레스"

    # 3. 선호 맛
    # 3~9. 맛 / 향 / 베이스는 절 단위로 긍정/부정 판별
    preferred_tastes = []
    disliked_tastes = []
    preferred_aromas = []
    disliked_aromas = []
    disliked_bases = []

    clauses = split_clauses(text)

    for clause in clauses:
        neg = is_negative_clause(clause)

        taste_tags = collect_tags(clause, TASTE_KEYWORDS)
        aroma_tags = collect_tags(clause, AROMA_KEYWORDS)
        base_tags = collect_tags(clause, BASE_KEYWORDS)

        if neg:
            disliked_tastes = _merge_list_slot(disliked_tastes, taste_tags)
            disliked_aromas = _merge_list_slot(disliked_aromas, aroma_tags)
            disliked_bases = _merge_list_slot(disliked_bases, base_tags)
        else:
            preferred_tastes = _merge_list_slot(preferred_tastes, taste_tags)
            preferred_aromas = _merge_list_slot(preferred_aromas, aroma_tags)

    # 비선호가 우선
    preferred_tastes = [x for x in preferred_tastes if x not in disliked_tastes]
    preferred_aromas = [x for x in preferred_aromas if x not in disliked_aromas]

    if preferred_tastes:
        extracted["preferred_tastes"] = _merge_list_slot(
            current_slots.get("preferred_tastes"),
            preferred_tastes,
        )

    if disliked_tastes:
        extracted["disliked_tastes"] = _merge_list_slot(
            current_slots.get("disliked_tastes"),
            disliked_tastes,
        )

    if preferred_aromas:
        extracted["preferred_aromas"] = _merge_list_slot(
            current_slots.get("preferred_aromas"),
            preferred_aromas,
        )

    if disliked_aromas:
        extracted["disliked_aromas"] = _merge_list_slot(
            current_slots.get("disliked_aromas"),
            disliked_aromas,
        )

    if disliked_bases:
        extracted["disliked_bases"] = _merge_list_slot(
            current_slots.get("disliked_bases"),
            disliked_bases,
        )

    # 10. 끝맛 선호
    if "깔끔한 끝맛" in text or "깔끔했으면" in text:
        extracted["finish_preference"] = "깔끔함"
    elif "달콤한 여운" in text:
        extracted["finish_preference"] = "달콤함"
    elif "가벼운 끝맛" in text:
        extracted["finish_preference"] = "가벼움"
        
    if "preferred_aromas" in extracted and "disliked_aromas" in extracted:
        extracted["preferred_aromas"] = [
            x for x in extracted["preferred_aromas"]
            if x not in extracted["disliked_aromas"]
        ]

    if "preferred_tastes" in extracted and "disliked_tastes" in extracted:
        extracted["preferred_tastes"] = [
            x for x in extracted["preferred_tastes"]
            if x not in extracted["disliked_tastes"]
        ]

    return extracted


def choose_next_question(slots: dict) -> str:
    question_map = {
        "party_purpose": "오늘은 어떤 자리에서 마시는 건가요? 예를 들면 생일파티, 친구 모임, 데이트 같은 식으로 알려주세요.",
        "current_mood": "지금 기분은 어떠세요? 신나는지, 차분한지, 기분 전환이 필요한지도 괜찮아요.",
        "preferred_tastes": "좋아하는 맛을 조금 더 알려주세요. 상큼한 맛, 달콤한 맛, 청량한 맛 중 어떤 쪽이 좋나요?",
        "disliked_tastes": "반대로 피하고 싶은 맛도 있나요? 예를 들면 너무 단맛, 너무 쓴맛 같은 느낌이요.",
        "preferred_aromas": "좋아하는 향이 있나요? 과일향, 민트향, 우디향처럼 편하게 말씀해주셔도 됩니다.",
        "disliked_aromas": "싫어하거나 피하고 싶은 향도 있나요?",
        "strength_preference": "도수는 어느 정도를 원하세요? 약한 편, 중간, 강한 편 중에서 골라도 괜찮아요.",
        "favorite_drinks": "평소에 좋아하는 술이나 칵테일이 있다면 말씀해주세요.",
        "disliked_bases": "싫어하는 베이스 술이 있나요? 예를 들면 위스키, 럼, 진 같은 종류요.",
        "finish_preference": "끝맛은 어떤 느낌이 좋으세요? 깔끔한 쪽인지, 달콤하게 남는 쪽인지 알려주세요.",
    }

    for slot in REQUIRED_SLOTS:
        if not _is_filled(slots.get(slot)):
            return question_map[slot]

    return "정보가 충분히 모였습니다. 이제 추천 단계로 넘어갈게요."


def run_dialogue(history: list, slots: dict, user_msg: str) -> dict:
    extracted = extract_slots_from_korean_text(user_msg, slots)

    updated_slots = dict(slots)
    updated_slots.update(extracted)

    completion = get_completion(updated_slots)
    should_proceed = completion >= 80

    question = (
        "정보가 충분히 모였습니다. 이제 추천 단계로 진행할게요."
        if should_proceed
        else choose_next_question(updated_slots)
    )

    return {
        "question": question,
        "extracted_slots": extracted,
        "updated_slots": updated_slots,
        "completion": completion,
        "should_proceed": should_proceed,
    }


def classify_intent(feedback_text: str) -> str:
    text = feedback_text.strip()

    if "이걸로" in text or "좋아" in text or "괜찮아" in text:
        return "ACCEPT"
    if "다른" in text or "바꿔" in text or "새로운" in text:
        return "REJECT"
    return "ADJUST"


def update_vector(current: dict, feedback_text: str) -> dict:
    new_vec = dict(current)
    text = feedback_text.strip()

    def get_score(key: str, default: float = 3.0) -> float:
        return float(new_vec.get(key, default))

    if "너무 달" in text or "덜 달" in text:
        new_vec["sweetness_score"] = max(1.0, get_score("sweetness_score") - 1.0)

    if "더 달" in text:
        new_vec["sweetness_score"] = min(5.0, get_score("sweetness_score") + 1.0)

    if "더 상큼" in text:
        new_vec["freshness_score"] = min(5.0, get_score("freshness_score") + 1.0)
        new_vec["sourness_score"] = min(5.0, get_score("sourness_score") + 1.0)

    if "덜 상큼" in text:
        new_vec["freshness_score"] = max(1.0, get_score("freshness_score") - 1.0)
        new_vec["sourness_score"] = max(1.0, get_score("sourness_score") - 1.0)

    if "더 쓴" in text or "더 쌉쌀" in text:
        new_vec["bitterness_score"] = min(5.0, get_score("bitterness_score") + 1.0)

    if "너무 쓰" in text:
        new_vec["bitterness_score"] = max(1.0, get_score("bitterness_score") - 1.0)

    if "무거워" in text:
        new_vec["body_score"] = max(1.0, get_score("body_score") - 1.0)

    if "더 묵직" in text:
        new_vec["body_score"] = min(5.0, get_score("body_score") + 1.0)

    return new_vec