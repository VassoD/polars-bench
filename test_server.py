"""
Quick integration test against the running server.
Usage: python test_server.py [--url http://localhost:8000]
"""
import argparse
import json
import sys

import httpx

parser = argparse.ArgumentParser()
parser.add_argument("--url", default="http://localhost:8000")
args = parser.parse_args()
BASE = args.url.rstrip("/")


def check_health():
    r = httpx.get(f"{BASE}/health", timeout=5)
    r.raise_for_status()
    print(f"health: {r.json()}")


def compare(actual, expected, type_hint):
    if actual is None:
        return False
    if type_hint == "float":
        try:
            return abs(float(actual) - float(expected)) < 1e-2
        except (TypeError, ValueError):
            return False
    return actual == expected


def run_eval():
    with open("data/eval_set.json") as f:
        cases = json.load(f)

    correct = 0
    for case in cases:
        payload = {
            "question_id": case["id"],
            "message": case["question"],
            "schema": case["schema"],
            "data_path": "data/sales.parquet",
        }
        r = httpx.post(f"{BASE}/chat", json=payload, timeout=60)
        if r.status_code != 200:
            print(f"✗ {case['id']} HTTP {r.status_code}: {r.text[:120]}")
            continue

        response = r.json()["response"]
        ok = compare(response, case["expected"], case.get("type", "scalar"))
        correct += int(ok)
        mark = "✓" if ok else "✗"
        print(f"{mark} {case['id']}: got={response!r}  expected={case['expected']!r}")

    print(f"\n{correct}/{len(cases)} correct")
    return correct == len(cases)


check_health()
ok = run_eval()
sys.exit(0 if ok else 1)
