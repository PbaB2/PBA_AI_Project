from __future__ import annotations

from typing import Optional
from sqlalchemy.orm import Session

from app.db.models import Cocktail, Recipe, Ingredient
from app.db.crud import (
    get_guest_session,
    get_initial_tag_response,
    get_preference_slot,
    get_preference_vector,
    get_latest_space_analysis_by_party,
    create_sample_recommendation,
    get_all_recipes_with_ingredients,
)

# ============================================================
# 매핑 테이블
# ============================================================

TASTE_TO_COCKTAIL = {
    "단맛":   "sweet_level",
    "달콤함": "sweet_level",
    "쓴맛":   "bitter_level",
    "쌉쌀함": "bitter_level",
    "신맛":   "sour_level",
    "상큼함": "sour_level",
    "청량함": "freshness_level",
    "바디감": "body_level",
    "크리미함": "creamy_level",
    "스파이시": "spicy_level",
    "고소함": "nutty_level",
}

AROMA_TO_INGREDIENT = {
    "민트향":    "minty_score",
    "과일향":    "fruity_score",
    "시트러스향": "citrus_score",
    "허브향":    "herbal_score",
    "커피향":    "coffee_score",
    "우디향":    "woody_score",
    "꽃향":     "floral_score",
}

STRENGTH_RANGE = {
    "약함": (0.0, 2.0),
    "중간": (1.5, 3.5),
    "강함": (3.0, 5.0),
}

# ======helper함수 추가=======
def _merge_unique_list(*values) -> list[str]:
    merged: list[str] = []
    for v in values:
        if not v:
            continue
        for item in v:
            if item not in merged:
                merged.append(item)
    return merged


def _normalize_strength_tag(value: Optional[str]) -> Optional[str]:
    if not value:
        return None

    mapping = {
        "약함": "약함",
        "약한": "약함",
        "술 못 마셔요": "약함",
        "가볍게": "약함",

        "중간": "중간",
        "적당히": "중간",
        "보통": "중간",
        "무난하게": "중간",

        "강함": "강함",
        "센 거": "강함",
        "센걸 원해요": "강함",
        "술 잘 마셔요": "강함",
    }
    return mapping.get(value, value)


def _calc_effective_completion(merged_slots: dict) -> float:
    fields = [
        merged_slots.get("party_purpose"),
        merged_slots.get("current_mood"),
        merged_slots.get("preferred_tastes"),
        merged_slots.get("disliked_tastes"),
        merged_slots.get("preferred_aromas"),
        merged_slots.get("disliked_aromas"),
        merged_slots.get("strength_preference"),
        merged_slots.get("favorite_drinks"),
        merged_slots.get("disliked_bases"),
        merged_slots.get("finish_preference"),
    ]

    filled = 0
    for value in fields:
        if value is None:
            continue
        if isinstance(value, list) and len(value) == 0:
            continue
        if isinstance(value, str) and value.strip() == "":
            continue
        filled += 1

    return round((filled / 10) * 100, 2)


# ============================================================
# 1. 사용자 프로필 빌드
# ============================================================

def build_user_profile(db: Session, guest_session_id: str) -> dict:
    guest   = get_guest_session(db, guest_session_id)
    tags    = get_initial_tag_response(db, guest_session_id)
    slot    = get_preference_slot(db, guest_session_id)
    vector  = get_preference_vector(db, guest_session_id)
    space   = get_latest_space_analysis_by_party(db, guest.party_session_id) if guest else None

    merged_slots = {
        "party_purpose": slot.party_purpose if slot else None,
        "current_mood": slot.current_mood if slot else None,
        "preferred_tastes": _merge_unique_list(
            tags.taste_tags_json if tags else [],
            slot.preferred_tastes_json if slot else [],
        ),
        "disliked_tastes": _merge_unique_list(
            slot.disliked_tastes_json if slot else [],
        ),
        "preferred_aromas": _merge_unique_list(
            tags.aroma_tags_json if tags else [],
            slot.preferred_aromas_json if slot else [],
        ),
        "disliked_aromas": _merge_unique_list(
            slot.disliked_aromas_json if slot else [],
        ),
        "strength_preference": (
            slot.strength_preference
            if slot and slot.strength_preference
            else _normalize_strength_tag(tags.strength_tag if tags else None)
        ),
        "favorite_drinks": slot.favorite_drinks_text if slot else None,
        "disliked_bases": _merge_unique_list(
            slot.disliked_bases_json if slot else [],
        ),
        "finish_preference": slot.finish_preference if slot else None,
    }

    effective_completion = _calc_effective_completion(merged_slots)

    return {
        "guest": guest,
        "initial_tags": tags,
        "slot": slot,
        "vector": vector,
        "space": space,
        "merged_slots": merged_slots,
        "effective_completion": effective_completion,
    }


# ============================================================
# 2. 후보 칵테일 조회
# ============================================================

def get_candidate_cocktails(db: Session, exclude_ids: list[int] = []) -> list[Cocktail]:
    query = db.query(Cocktail).filter(Cocktail.is_active == True)
    if exclude_ids:
        query = query.filter(Cocktail.cocktail_id.notin_(exclude_ids))
    return query.all()


# ============================================================
# 3. 비선호 베이스 
# ============================================================

def _has_disliked_base(merged_slots: dict, recipe_ingredients: list[tuple]) -> bool:
    disliked_bases = merged_slots.get("disliked_bases") or []
    if not disliked_bases:
        return False

    for _, ingredient in recipe_ingredients:
        if ingredient.ingredient_type == "BASE" and ingredient.ingredients_name in disliked_bases:
            return True
    return False


# ============================================================
# 4. 칵테일 점수 계산
# ============================================================

def _normalize(user: float, cocktail: float, max_range: float = 4.0) -> float:
    """두 점수 차이를 0~1 유사도로 변환"""
    return 1.0 - abs(user - cocktail) / max_range


def score_cocktail(
    cocktail: Cocktail,
    profile: dict,
    recipe_ingredients: list[tuple],
) -> float:
    merged = profile["merged_slots"]
    vector = profile["vector"]
    space  = profile["space"]
    score  = 0.0

    # 1. 벡터 유사도
    pairs = [
        (vector.sweetness_score,  cocktail.sweet_level,     4.0, 20),
        (vector.sourness_score,   cocktail.sour_level,      4.0, 15),
        (vector.bitterness_score, cocktail.bitter_level,    4.0, 10),
        (vector.freshness_score,  cocktail.freshness_level, 4.0, 10),
        (vector.body_score,       cocktail.body_level,      4.0,  8),
    ]
    for user_s, cocktail_s, max_r, weight in pairs:
        if cocktail_s is None:
            continue
        score += _normalize(float(user_s), float(cocktail_s), max_r) * weight

    # 2. 선호 맛 보너스
    for tag in (merged.get("preferred_tastes") or []):
        col = TASTE_TO_COCKTAIL.get(tag)
        if col and getattr(cocktail, col) is not None:
            if float(getattr(cocktail, col)) >= 3.5:
                score += 10

    # 3. 선호 향 보너스
    for _, ingredient in recipe_ingredients:
        for tag in (merged.get("preferred_aromas") or []):
            col = AROMA_TO_INGREDIENT.get(tag)
            if col and getattr(ingredient, col, 0) >= 3.0:
                score += 8

    # 4. 공간 무드 보너스
    if space and cocktail.mood_tag:
        mood_prob = (space.mood_tags_json or {}).get(cocktail.mood_tag, 0.0)
        score += mood_prob * 15

    # 5. 비선호 맛 패널티
    for tag in (merged.get("disliked_tastes") or []):
        col = TASTE_TO_COCKTAIL.get(tag)
        if col and getattr(cocktail, col) is not None:
            val = float(getattr(cocktail, col))
            if val >= 4.0:
                score -= 25
            elif val >= 3.0:
                score -= 10

    # 6. 비선호 향 패널티
    for _, ingredient in recipe_ingredients:
        for tag in (merged.get("disliked_aromas") or []):
            col = AROMA_TO_INGREDIENT.get(tag)
            if col and getattr(ingredient, col, 0) >= 3.0:
                score -= 20

    # 7. 도수 선호 보너스
    strength_pref = merged.get("strength_preference")
    if strength_pref in STRENGTH_RANGE:
        low, high = STRENGTH_RANGE[strength_pref]
        if low <= float(vector.alcohol_score) <= high:
            score += 10

    return round(score, 2)


# ============================================================
# 5. Top-K 추천
# ============================================================

def recommend_top_k(
    db: Session,
    guest_session_id: str,
    k: int = 3,
    exclude_ids: list[int] = [],
) -> list[dict]:
    profile      = build_user_profile(db, guest_session_id)
    merged_slots = profile["merged_slots"]
    candidates   = get_candidate_cocktails(db, exclude_ids=exclude_ids)

    all_ri = get_all_recipes_with_ingredients(db)

    results = []
    for cocktail in candidates:
        ri = all_ri.get(cocktail.cocktail_id, [])

        if _has_disliked_base(merged_slots, ri):
            continue

        s = score_cocktail(cocktail, profile, ri)
        results.append({
            "cocktail_id": cocktail.cocktail_id,
            "name_kr": cocktail.name_kr,
            "score": s,
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:k]


# ============================================================
# 6. 추천 실행 + DB 저장
# ============================================================

def run_recommendation(
    db: Session,
    guest_session_id: str,
    k: int = 3,
    exclude_ids: list[int] = [],
) -> dict:
    profile = build_user_profile(db, guest_session_id)
    space   = profile["space"]

    if not space:
        return {
            "status": "need_space_image",
            "message": "공간 분석 결과가 없습니다. 이미지를 먼저 업로드해주세요."
        }

    effective_completion = profile["effective_completion"]
    if effective_completion < 80:
        return {
            "status": "need_more_info",
            "completion": effective_completion,
            "message": "아직 추천에 필요한 정보가 부족합니다."
        }

    top_k = recommend_top_k(db, guest_session_id, k=k, exclude_ids=exclude_ids)

    if not top_k:
        return {
            "status": "no_candidates",
            "top_k": [],
            "sample_recommendation_id": None,
        }

    best = top_k[0]

    row = create_sample_recommendation(
        db=db,
        guest_session_id=guest_session_id,
        space_analysis_id=space.space_analysis_id,
        recommended_cocktail_id=best["cocktail_id"],
        recommendation_reason=f"취향 점수 기반 추천 (score: {best['score']})",
        rag_retrieved_ids_json=[r["cocktail_id"] for r in top_k],
        recipe_snapshot_json={"top_k": top_k},
    )

    return {
        "status": "ok",
        "completion": effective_completion,
        "top_k": top_k,
        "sample_recommendation_id": str(row.sample_recommendation_id),
    }
