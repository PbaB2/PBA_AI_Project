import json
import os
from pathlib import Path

import requests
from PIL import Image

BASE = os.getenv("BASE_URL", "http://141.223.140.32:8000/api/v1")


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
        json={"session_name": "strict smoke test party", "demo_mode": True},
        timeout=20,
    )
    party_data = expect_ok(r, "party")
    party_id = party_data["party_session_id"]

    # 2. guest
    r = requests.post(
        f"{BASE}/sessions/guest",
        json={"party_session_id": party_id, "guest_label": "게스트1"},
        timeout=20,
    )
    guest_data = expect_ok(r, "guest")
    guest_id = guest_data["guest_session_id"]

    # 3. initial tags
    r = requests.post(
        f"{BASE}/sessions/{guest_id}/tags",
        json={
            "familiarity": "가끔 마셔요",
            "tastes": ["단맛", "청량감"],
            "strength": "적당히",
            "aromas": ["과일향"],
        },
        timeout=20,
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
            timeout=20,
        )
    expect_ok(r, "space-image")

    # 5. LLM first question
    r = requests.post(f"{BASE}/sessions/{guest_id}/start-dialogue", timeout=20)
    start_data = expect_ok(r, "start-dialogue")
    print("첫 질문:", start_data.get("question"))

    # 6. strict slot-filling dialogue
    messages = [
        "생일파티예요",
        "기분이 신나요",
        "단맛이 좋아요",
        "위스키는 싫어요",
        "적당히 마시고 싶어요",
        "모히토를 좋아해요",
        "끝맛은 깔끔하게 좋겠어요",
        "과일향 좋아요"
    ]

    last_dialogue = None

    for idx, msg in enumerate(messages, start=1):
        print(f"\n[USER-{idx}] {msg}")
        r = requests.post(
            f"{BASE}/sessions/{guest_id}/dialogue",
            json={"message": msg},
            timeout=20,
        )
        last_dialogue = expect_ok(r, f"dialogue-{idx}")

        status = last_dialogue.get("status")
        if status in ["proceed_to_recommendation", "max_turns_reached"]:
            break
        if last_dialogue.get("should_proceed", False):
            break

    if last_dialogue is None:
        raise RuntimeError("dialogue result is empty")

    print("\n=== dialogue summary ===")
    print("status:", last_dialogue.get("status"))
    print("completion:", last_dialogue.get("completion"))
    print("should_proceed:", last_dialogue.get("should_proceed"))

    if last_dialogue.get("status") not in ["proceed_to_recommendation", "max_turns_reached"] \
       and not last_dialogue.get("should_proceed", False):
        raise RuntimeError(
            "strict smoke test failed: 80% completion or proceed condition not reached"
        )

    # 7. recommendation (strict: no force)
    r = requests.post(
        f"{BASE}/sessions/{guest_id}/recommend-sample",
        timeout=20,
    )
    rec_data = expect_ok(r, "recommend-sample")

    if rec_data.get("status") != "ok":
        raise RuntimeError(f"recommendation failed in strict mode: {rec_data}")

    print("\n=== strict recommendation result ===")
    print(json.dumps(rec_data, ensure_ascii=False, indent=2))

    print("\n[PASS] strict slot smoke test completed")


if __name__ == "__main__":
    main()