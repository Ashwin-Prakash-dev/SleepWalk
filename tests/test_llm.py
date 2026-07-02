import json
from llm_service import extract_node, run_inference

# ── Test 1: extract a fact ──────────────────────────────────────
print("TEST 1: fact extraction")
result = extract_node(
    "The FIFA World Cup 2022 was held in Qatar."
)
assert result["node_kind"] == "fact"
print(json.dumps(result, indent=2))

assert result["node_kind"] == "fact", f"expected fact, got {result['node_kind']}"
assert result["confidence"] > 0.85, f"confidence too low: {result['confidence']}"
assert result["subject"] is not None
print("PASS\n")

# ── Test 2: extract a claim ─────────────────────────────────────
print("TEST 2: claim extraction")
result = extract_node(
    "Iran's foreign minister stated that demands to reduce its missile "
    "program are completely unacceptable and violate Iranian sovereignty."
)
print(json.dumps(result, indent=2))

assert result["node_kind"] in ("claim", "position"), f"got {result['node_kind']}"
assert result["actor"] is not None, "actor should not be null"
assert "iran" in result["actor"].lower(), f"actor should be Iran, got {result['actor']}"
print("PASS\n")

# ── Test 3: extract a denial ────────────────────────────────────
print("TEST 3: denial extraction")
result = extract_node(
    "Iran denied that it had agreed to any limits on its ballistic missile capabilities."
)
print(json.dumps(result, indent=2))

assert result["node_kind"] == "denial", f"expected denial, got {result['node_kind']}"
assert result["denies_claim"] is not None, "denies_claim should be populated"
print("PASS\n")

# ── Test 4: extract an event announcement ──────────────────────
print("TEST 4: event announcement extraction")
result = extract_node(
    "The UN Security Council emergency meeting on Iran sanctions is scheduled for March 15, 2026."
)
print(json.dumps(result, indent=2))

assert result["node_kind"] == "event_announcement"
assert result["expires_at"] is not None, "expires_at should be set for announcements"
print("PASS\n")

# ── Test 5: run inference against contradicting nodes ───────────
print("TEST 5: contradiction inference")
new_node = {
    "actor": "United States",
    "node_kind": "claim",
    "subject": "Iran missile deal",
    "content": "The US announced it is close to a deal with Iran that includes reducing its missile capabilities."
}
similar_nodes = [
    {
        "index": 0,
        "actor": "Iran",
        "node_kind": "position",
        "subject": "missile capabilities",
        "content": "Iran declared that any demand to reduce its missile program is an impossible red line.",
        "confidence": 0.95
    },
    {
        "index": 1,
        "actor": "Israel",
        "node_kind": "demand",
        "subject": "missile capabilities",
        "content": "Israel stated that missile reduction must be part of any Iran nuclear deal.",
        "confidence": 0.90
    }
]
inferences = run_inference(new_node, similar_nodes)
print(json.dumps(inferences, indent=2))

assert isinstance(inferences, list), "should return a list"
assert len(inferences) > 0, "should find at least one inference"

kinds = [i["inference_kind"] for i in inferences]
assert "contradiction" in kinds, f"should detect contradiction, got: {kinds}"

for inf in inferences:
    assert "content" in inf
    assert "confidence" in inf
    assert 0.0 <= inf["confidence"] <= 1.0
    assert "source_node_indices" in inf
print("PASS\n")

# ── Test 6: JSON robustness ─────────────────────────────────────
print("TEST 6: handles edge case input")
result = extract_node("Apple reported record quarterly earnings of $94 billion.")
print(json.dumps(result, indent=2))
assert result["node_kind"] is not None
assert result["content"] is not None
print("PASS\n")

print("=" * 40)
print("All tests passed. Phase 2 is good.")