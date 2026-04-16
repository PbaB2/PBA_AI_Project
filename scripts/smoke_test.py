# scripts/smoke_test.py
import requests
import json
from pathlib import Path
from PIL import Image

BASE = "http://localhost:8000/api/v1"


def expect_ok(resp, step_name):
    print(f"[{step_name}] status={resp.status_code}")

    try:
        data = resp.json()
        print(json.dumps(data, ensure_ascii=False, indent=2))
    except Exception:
        print(resp.text)
        raise RuntimeError(f"{step_name} returned non-JSON response")

    if not resp.ok:
        raise RuntimeError(f"{step_name} failed: {resp.status_code}")

    return data


def main():
    # 1. party
    r = requests.post(
        f"{BASE}/sessions/party",
        json={"session_name": "smoke test party", "demo_mode": True},
    )
    party_data = expect_ok(r, "party")
    party_id = party_data["party_session_id"]

    # 2. guest
    r = requests.post(
        f"{BASE}/sessions/guest",
        json={"party_session_id": party_id, "guest_label": "게스트1"},
    )
    guest_data = expect_ok(r, "guest")
    guest_id = guest_data["guest_session_id"]

    # 3. tags
    r = requests.post(
        f"{BASE}/sessions/{guest_id}/tags",
        json={
            "familiarity": "가끔 마셔요",
            "tastes": ["단맛", "청량감"],
            "strength": "적당히",
            "aromas": ["과일향"],
        },
    )
    expect_ok(r, "tags")

    # 4. image
    image_path = Path("data/test.jpg")
    image_path.parent.mkdir(parents=True, exist_ok=True)
    if not image_path.exists():
        img = Image.new("RGB", (256, 256), color=(255, 180, 120))
        img.save(image_path)

    with open(image_path, "rb") as f:
        r = requests.post(
            f"{BASE}/sessions/{guest_id}/space-image",
            files={"file": ("test.jpg", f, "image/jpeg")},
        )
    expect_ok(r, "space-image")

    # 5. dialogue turns
    messages = [
        "생일파티인데 달콤한 게 좋아요",
        "기분은 엄청 좋고 과일향 나는 칵테일이면 좋겠어요",
        "도수는 적당한 편이 좋고 민트향은 싫어요. 끝맛은 깔끔했으면 좋겠어요",
    ]

    last_dialogue = None
    for idx, msg in enumerate(messages, start=1):
        r = requests.post(
            f"{BASE}/sessions/{guest_id}/dialogue",
            json={"message": msg},
        )
        last_dialogue = expect_ok(r, f"dialogue-{idx}")

    print("마지막 질문:", last_dialogue.get("question"))
    print("completion:", last_dialogue.get("completion"))
    print("should_proceed:", last_dialogue.get("should_proceed"))

    # 6. recommendation
    if not last_dialogue.get("should_proceed", False):
        print("아직 정보 부족해서 추천 호출 안 함")
        return

    r = requests.post(f"{BASE}/sessions/{guest_id}/recommend-sample")
    rec_data = expect_ok(r, "recommend-sample")
    print("추천 결과:", json.dumps(rec_data, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()