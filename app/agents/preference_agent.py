from __future__ import annotations

from typing import Any


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
    preferred_tastes = []
    if "상큼" in text:
        preferred_tastes.append("상큼함")
    if "달콤" in text or "단맛" in text:
        preferred_tastes.append("단맛")
    if "청량" in text:
        preferred_tastes.append("청량함")
    if "고소" in text:
        preferred_tastes.append("고소함")
    if "스파이시" in text or "매콤" in text:
        preferred_tastes.append("스파이시")
    if "밀키" in text or "부드러운" in text:
        preferred_tastes.append("밀키함")

    if preferred_tastes:
        extracted["preferred_tastes"] = _merge_list_slot(
            current_slots.get("preferred_tastes"),
            preferred_tastes,
        )

    # 4. 비선호 맛
    disliked_tastes = []
    if "쓴 맛 싫" in text or "너무 쓰" in text:
        disliked_tastes.append("쓴맛")
    if "너무 달" in text or "단 거 싫" in text:
        disliked_tastes.append("단맛")
    if "신 거 싫" in text:
        disliked_tastes.append("신맛")

    if disliked_tastes:
        extracted["disliked_tastes"] = _merge_list_slot(
            current_slots.get("disliked_tastes"),
            disliked_tastes,
        )

    # 5. 선호 향
    preferred_aromas = []
    if "민트" in text:
        preferred_aromas.append("민트향")
    if "과일향" in text or "과일" in text:
        preferred_aromas.append("과일향")
    if "우디" in text:
        preferred_aromas.append("우디향")
    if "커피향" in text or "커피" in text:
        preferred_aromas.append("커피향")
    if "시트러스" in text:
        preferred_aromas.append("시트러스향")
    if "허브" in text:
        preferred_aromas.append("허브향")

    if preferred_aromas:
        extracted["preferred_aromas"] = _merge_list_slot(
            current_slots.get("preferred_aromas"),
            preferred_aromas,
        )

    # 6. 비선호 향
    disliked_aromas = []
    if "민트향 싫" in text:
        disliked_aromas.append("민트향")
    if "커피향 싫" in text:
        disliked_aromas.append("커피향")
    if "우디향 싫" in text:
        disliked_aromas.append("우디향")

    if disliked_aromas:
        extracted["disliked_aromas"] = _merge_list_slot(
            current_slots.get("disliked_aromas"),
            disliked_aromas,
        )

    # 7. 도수 선호
    if "술 잘 마셔" in text or "센 거 좋아" in text or "도수 높은" in text:
        extracted["strength_preference"] = "강함"
    elif "술 못 마셔" in text or "약한 거" in text or "도수 낮은" in text:
        extracted["strength_preference"] = "약함"
    elif "적당한" in text or "중간 정도" in text:
        extracted["strength_preference"] = "중간"

    # 8. 평소 좋아하는 음료
    if "모히토" in text:
        extracted["favorite_drinks"] = "모히토"
    elif "하이볼" in text:
        extracted["favorite_drinks"] = "하이볼"
    elif "마가리타" in text:
        extracted["favorite_drinks"] = "마가리타"

    # 9. 싫어하는 베이스
    disliked_bases = []
    if "위스키 싫" in text:
        disliked_bases.append("위스키")
    if "럼 싫" in text:
        disliked_bases.append("럼")
    if "진 싫" in text:
        disliked_bases.append("진")
    if "데킬라 싫" in text:
        disliked_bases.append("데킬라")
    if "보드카 싫" in text:
        disliked_bases.append("보드카")

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