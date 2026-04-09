"""
test_api.py — Run this to test the API after starting it with uvicorn
Make sure uvicorn is running first:
    uvicorn main:app --reload
Then run:
    python test_api.py
"""

import requests
import json

BASE_URL = "http://127.0.0.1:8000"

# ── Test complaints ───────────────────────────────────────
test_cases = [
    {
        "text": "There is a large pothole on the main road causing damage to vehicles.",
        "expected_dept": "Roads & Transport"
    },
    {
        "text": "Garbage has not been collected for 5 days. Terrible smell everywhere.",
        "expected_dept": "Sanitation"
    },
    {
        "text": "Street light is broken near the park. Very dangerous at night.",
        "expected_dept": "Roads & Transport"
    },
    {
        "text": "Dead animal lying on the sidewalk for 3 days. Health hazard.",
        "expected_dept": "Sanitation"
    },
    {
        "text": "Loud music party next door every night, residents cannot sleep.",
        "expected_dept": "Health & Sanitation"
    },
]

print("=" * 70)
print("  Citizen Grievance NLP API — Test Results")
print("=" * 70)

# Health check
print("\n[1] Health Check:")
resp = requests.get(f"{BASE_URL}/health")
print(f"    Status: {resp.status_code}")
print(f"    Response: {resp.json()}")

# Predictions
print("\n[2] Prediction Tests:")
print("-" * 70)

for i, tc in enumerate(test_cases, 1):
    resp = requests.post(
        f"{BASE_URL}/predict",
        json={"text": tc["text"]}
    )

    if resp.status_code == 200:
        r = resp.json()
        print(f"\nTest {i}:")
        print(f"  Complaint:      {tc['text'][:60]}...")
        print(f"  Department:     {r['department']}")
        print(f"  Sentiment:      {r['sentiment']}")
        print(f"  Priority Score: {r['priority_score']}")
        print(f"  Priority Label: {r['priority_label']}")
        print(f"  Confidence:     {r['confidence']:.3f}")
    else:
        print(f"\nTest {i} FAILED: {resp.status_code} — {resp.text}")

print("\n" + "=" * 70)
print("  All tests complete!")
print("  Visit http://127.0.0.1:8000/docs for interactive Swagger UI")
print("=" * 70)
