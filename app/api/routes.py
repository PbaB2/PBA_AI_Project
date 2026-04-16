import os
import shutil
from typing import List, Optional, Any

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.crud import (
    create_party_session,
    create_guest_session,
    get_guest_session,
    upsert_initial_tags,
    save_space_analysis,
    create_dialogue_turn,
    list_dialogue_turns,
    get_preference_slot,
    update_preference_slots,
    get_preference_vector,
    create_sample_feedback,
    update_preference_vector,
    get_latest_sample_recommendation_by_guest,
)
from app.utils.config import UPLOAD_DIR
from app.agents.preference_agent import run_dialogue, classify_intent, update_vector
from app.agents.orchestration_agent import run_recommendation

router = APIRouter()


# ============================================================
# Pydantic Schemas
# ============================================================

class PartySessionCreateRequest(BaseModel):
    session_name: Optional[str] = None
    demo_mode: bool = False


class GuestSessionCreateRequest(BaseModel):
    party_session_id: str
    guest_label: str = "게스트1"


class InitialTagRequest(BaseModel):
    familiarity: Optional[str] = Field(default=None, description="칵테일 친숙도")
    tastes: List[str] = Field(default_factory=list, description="초기 선호 맛 태그")
    strength: str = Field(..., description="초기 선호 도수/강도감 태그")
    aromas: List[str] = Field(default_factory=list, description="초기 선호 향 태그")


class DialogueRequest(BaseModel):
    message: str


class FeedbackRequest(BaseModel):
    sample_recommendation_id: str | None = None
    feedback_text: str


# ============================================================
# Helpers
# ============================================================

def _serialize_dialogue_turn(turn) -> dict:
    return {
        "turn_id": str(turn.turn_id),
        "guest_session_id": str(turn.guest_session_id),
        "turn_index": turn.turn_index,
        "speaker_role": turn.speaker_role,
        "utterance_text": turn.utterance_text,
        "extracted_slots_json": turn.extracted_slots_json,
        "created_at": turn.created_at.isoformat() if turn.created_at else None,
    }


def _slot_row_to_internal_dict(slot_row) -> dict:
    if slot_row is None:
        return {}

    return {
        "party_purpose": slot_row.party_purpose,
        "current_mood": slot_row.current_mood,
        "preferred_tastes": slot_row.preferred_tastes_json,
        "disliked_tastes": slot_row.disliked_tastes_json,
        "preferred_aromas": slot_row.preferred_aromas_json,
        "disliked_aromas": slot_row.disliked_aromas_json,
        "strength_preference": slot_row.strength_preference,
        "favorite_drinks": slot_row.favorite_drinks_text,
        "disliked_bases": slot_row.disliked_bases_json,
        "finish_preference": slot_row.finish_preference,
    }


def _serialize_preference_slot(slot_row) -> dict:
    return {
        "slot_profile_id": str(slot_row.slot_profile_id),
        "guest_session_id": str(slot_row.guest_session_id),
        "party_purpose": slot_row.party_purpose,
        "current_mood": slot_row.current_mood,
        "preferred_tastes_json": slot_row.preferred_tastes_json,
        "disliked_tastes_json": slot_row.disliked_tastes_json,
        "preferred_aromas_json": slot_row.preferred_aromas_json,
        "disliked_aromas_json": slot_row.disliked_aromas_json,
        "strength_preference": slot_row.strength_preference,
        "favorite_drinks_text": slot_row.favorite_drinks_text,
        "disliked_bases_json": slot_row.disliked_bases_json,
        "finish_preference": slot_row.finish_preference,
        "slot_completion_score": float(slot_row.slot_completion_score),
        "updated_at": slot_row.updated_at.isoformat() if slot_row.updated_at else None,
    }


# ============================================================
# Basic
# ============================================================

@router.get("/health")
def healthcheck():
    return {"status": "ok"}


# ============================================================
# 추천
# ============================================================

@router.post("/sessions/{gid}/recommend-sample")
def recommend_sample_endpoint(
    gid: str,
    db: Session = Depends(get_db),
):
    guest = get_guest_session(db, gid)
    if not guest:
        raise HTTPException(status_code=404, detail="guest_session_id not found")

    result = run_recommendation(db, gid, k=3)

    # 1) 공간 이미지 없음
    if result.get("status") == "need_space_image":
        return result

    # 2) 정보 부족
    if result.get("status") == "need_more_info":
        return result

    # 3) 후보 없음
    if result.get("status") == "no_candidates":
        return result

    # 4) 정상 추천
    return result


@router.post("/sessions/{gid}/feedback")
def feedback_endpoint(
    gid: str,
    req: FeedbackRequest,
    db: Session = Depends(get_db),
):
    guest = get_guest_session(db, gid)
    if not guest:
        raise HTTPException(status_code=404, detail="guest_session_id not found")

    current_vector = get_preference_vector(db, gid)
    if not current_vector:
        raise HTTPException(status_code=404, detail="preference_vector not found")

    # sample_recommendation_id가 안 오면 가장 최근 추천 사용
    sample_recommendation = None
    if req.sample_recommendation_id:
        # 지금은 latest 기반으로 충분해서 별도 get 없이도 되지만,
        # 안전하게 latest만 써도 됨
        latest = get_latest_sample_recommendation_by_guest(db, gid)
        if not latest or str(latest.sample_recommendation_id) != req.sample_recommendation_id:
            sample_recommendation = latest
        else:
            sample_recommendation = latest
    else:
        sample_recommendation = get_latest_sample_recommendation_by_guest(db, gid)

    if not sample_recommendation:
        raise HTTPException(status_code=404, detail="sample_recommendation not found")

    intent = classify_intent(req.feedback_text)

    vector_dict = {
        "sweetness_score": float(current_vector.sweetness_score),
        "bitterness_score": float(current_vector.bitterness_score),
        "sourness_score": float(current_vector.sourness_score),
        "freshness_score": float(current_vector.freshness_score),
        "body_score": float(current_vector.body_score),
        "herbal_score": float(current_vector.herbal_score),
        "citrus_score": float(current_vector.citrus_score),
        "alcohol_score": float(current_vector.alcohol_score),
    }

    updated_vector_dict = update_vector(vector_dict, req.feedback_text)

    saved_feedback = create_sample_feedback(
        db=db,
        sample_recommendation_id=sample_recommendation.sample_recommendation_id,
        feedback_text=req.feedback_text,
        feedback_intent=intent,
        parsed_summary=f"분류 결과: {intent}",
    )

    updated_vector = update_preference_vector(
        db=db,
        guest_session_id=gid,
        updates=updated_vector_dict,
        increment_version=True,
    )

    return {
        "status": "ok",
        "sample_feedback_id": str(saved_feedback.sample_feedback_id),
        "sample_recommendation_id": str(sample_recommendation.sample_recommendation_id),
        "feedback_intent": intent,
        "feedback_text": saved_feedback.feedback_text,
        "updated_vector": {
            "sweetness_score": float(updated_vector.sweetness_score),
            "bitterness_score": float(updated_vector.bitterness_score),
            "sourness_score": float(updated_vector.sourness_score),
            "freshness_score": float(updated_vector.freshness_score),
            "body_score": float(updated_vector.body_score),
            "herbal_score": float(updated_vector.herbal_score),
            "citrus_score": float(updated_vector.citrus_score),
            "alcohol_score": float(updated_vector.alcohol_score),
            "version": updated_vector.version,
        },
        "created_at": saved_feedback.created_at.isoformat() if saved_feedback.created_at else None,
    }

# ============================================================
# 1. Party / Guest Session
# ============================================================

@router.post("/sessions/party")
def create_party(req: PartySessionCreateRequest, db: Session = Depends(get_db)):
    row = create_party_session(
        db=db,
        session_name=req.session_name,
        demo_mode=req.demo_mode,
    )

    return {
        "party_session_id": str(row.party_session_id),
        "session_name": row.session_name,
        "session_status": row.session_status,
        "started_at": row.started_at.isoformat() if row.started_at else None,
        "demo_mode": row.demo_mode,
    }


@router.post("/sessions/guest")
def create_guest(req: GuestSessionCreateRequest, db: Session = Depends(get_db)):
    row = create_guest_session(
        db=db,
        party_session_id=req.party_session_id,
        guest_label=req.guest_label,
    )

    return {
        "guest_session_id": str(row.guest_session_id),
        "party_session_id": str(row.party_session_id),
        "guest_label": row.guest_label,
        "session_status": row.session_status,
        "started_at": row.started_at.isoformat() if row.started_at else None,
    }


# ============================================================
# 2. Initial Tags
# ============================================================

@router.post("/sessions/{gid}/tags")
def save_initial_tags_endpoint(
    gid: str,
    req: InitialTagRequest,
    db: Session = Depends(get_db),
):
    guest = get_guest_session(db, gid)
    if not guest:
        raise HTTPException(status_code=404, detail="guest_session_id not found")

    row = upsert_initial_tags(
        db=db,
        guest_session_id=gid,
        familiarity_tag=req.familiarity,
        taste_tags_json=req.tastes,
        strength_tag=req.strength,
        aroma_tags_json=req.aromas,
    )

    return {
        "status": "ok",
        "tag_response_id": str(row.tag_response_id),
        "guest_session_id": str(row.guest_session_id),
        "familiarity_tag": row.familiarity_tag,
        "taste_tags_json": row.taste_tags_json,
        "strength_tag": row.strength_tag,
        "aroma_tags_json": row.aroma_tags_json,
        "created_at": row.created_at.isoformat() if row.created_at else None,
    }


# ============================================================
# 3. Space Image Upload
# ============================================================

@router.post("/sessions/{gid}/space-image")
async def upload_space_image(
    gid: str,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    guest = get_guest_session(db, gid)
    if not guest:
        raise HTTPException(status_code=404, detail="guest_session_id not found")

    filename = f"{gid}_{file.filename}"
    file_path = os.path.join(UPLOAD_DIR, filename)

    # 업로드 폴더 없으면 생성
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    file_path = os.path.join(UPLOAD_DIR, filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # TODO:
    # 나중에 분위기 파악 에이전트(Qwen2-VL 등) 연결 시
    # 아래 placeholder를 실제 분석 결과로 교체
    mood_tags_json = {
        "신나는": 0.91,
        "네온": 0.74,
    }

    analysis = save_space_analysis(
        db=db,
        party_session_id=guest.party_session_id,
        image_path=file_path,
        mood_tags_json=mood_tags_json,
        mood_weight=0.30,
    )

    return {
        "status": "ok",
        "space_analysis": {
            "space_analysis_id": str(analysis.space_analysis_id),
            "party_session_id": str(analysis.party_session_id),
            "guest_session_id": gid,
            "image_path": analysis.image_path,
            "mood_tags_json": analysis.mood_tags_json,
            "mood_weight": float(analysis.mood_weight),
            "created_at": analysis.created_at.isoformat() if analysis.created_at else None,
        },
    }


# ============================================================
# 4. Dialogue
# ============================================================

@router.post("/sessions/{gid}/dialogue")
def dialogue_endpoint(
    gid: str,
    req: DialogueRequest,
    db: Session = Depends(get_db),
):
    guest = get_guest_session(db, gid)
    if not guest:
        raise HTTPException(status_code=404, detail="guest_session_id not found")

    # 1) 사용자 턴 저장
    user_turn = create_dialogue_turn(
        db=db,
        guest_session_id=gid,
        speaker_role="USER",
        utterance_text=req.message,
        extracted_slots_json=None,
    )

    # 2) 현재 슬롯 상태 조회
    slot_row = get_preference_slot(db, gid)
    current_slots = _slot_row_to_internal_dict(slot_row)

    # 3) 전체 대화 히스토리 조회
    history_rows = list_dialogue_turns(db, gid)
    history = [
        {
            "speaker_role": row.speaker_role,
            "utterance_text": row.utterance_text,
            "extracted_slots_json": row.extracted_slots_json,
        }
        for row in history_rows
    ]

    # 4) 에이전트 실행
    agent_result = run_dialogue(
        history=history,
        slots=current_slots,
        user_msg=req.message,
    )

    extracted_slots = agent_result.get("extracted_slots", {})
    question = agent_result.get("question", "")
    completion = agent_result.get("completion", 0.0)
    should_proceed = agent_result.get("should_proceed", False)

    # 5) 추출된 슬롯 DB 반영
    updated_slot_row = update_preference_slots(
        db=db,
        guest_session_id=gid,
        extracted_slots=extracted_slots,
    )

    # 6) LLM 질문 턴 저장
    llm_turn = create_dialogue_turn(
        db=db,
        guest_session_id=gid,
        speaker_role="LLM",
        utterance_text=question,
        extracted_slots_json=extracted_slots,
    )

    return {
        "status": "ok",
        "guest_session_id": gid,
        "user_turn": _serialize_dialogue_turn(user_turn),
        "llm_turn": _serialize_dialogue_turn(llm_turn),
        "question": question,
        "extracted_slots": extracted_slots,
        "slot_state": _serialize_preference_slot(updated_slot_row),
        "completion": completion,
        "should_proceed": should_proceed,
    }