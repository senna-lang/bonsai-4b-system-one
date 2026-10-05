"""Weight-free checks of the System One request translation and answer shapes."""
import pytest

from system_one_bonsai.serve import to_answers, to_request

QUESTIONS = {
    "team": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "Payments", "tech": ""}},
    "urgency": {"type": "score", "instructions": "How urgent?", "criteria": ["low", "medium", "high"]},
    "refund": {"type": "noul", "instructions": "Refund requested?", "criteria": {"true": "Asks for money back", "false": "Does not"}},
}


def test_questions_become_branches_in_order_with_labels_kept():
    request, meta = to_request({"state": {"ticket": "Charged twice"}, "questions": QUESTIONS})
    assert [b.kind for b in request.branches] == ["choice", "score", "noul"]
    assert request.branches[0].options == ["billing: Payments", "tech"]
    assert request.branches[1].options == ["low", "medium", "high"]
    assert "Asks for money back" in request.branches[2].instructions and request.branches[2].options == []
    assert '"ticket": "Charged twice"' in request.state
    assert meta[0] == ("team", "choice", ["billing", "tech"])


def test_bool_type_is_accepted_as_noul():
    _, meta = to_request({"state": {"a": 1}, "questions": {"q": {"type": "bool", "instructions": "Yes?"}}})
    assert meta == [("q", "noul", ["No", "Yes"])]


def test_answers_have_the_shapes_clients_parse():
    _, meta = to_request({"state": {"ticket": "x"}, "questions": QUESTIONS})
    answers = to_answers(meta, [[0.2, 0.8], [0.1, 0.3, 0.6], [0.25, 0.75]])
    assert answers["team"] == {"type": "choice", "choice": "tech", "probabilities": {"billing": 0.2, "tech": 0.8}, "confidence": 0.8}
    assert answers["urgency"]["type"] == "score" and answers["urgency"]["score"] == pytest.approx(1.5)
    assert answers["refund"] == {"type": "noul", "noul": 0.75}


@pytest.mark.parametrize("body", [
    {"state": "text", "questions": QUESTIONS},
    {"state": {"a": 1}, "questions": {}},
    {"state": {"a": 1}, "questions": {"q": {"type": "choice", "instructions": "?", "criteria": ["a", "b"]}}},
    {"state": {"a": 1}, "questions": {"q": {"type": "choice", "instructions": "?", "criteria": {"only": ""}}}},
    {"state": {"a": 1}, "questions": {"q": {"type": "rank", "instructions": "?", "criteria": []}}},
])
def test_invalid_requests_are_rejected(body):
    with pytest.raises(ValueError):
        to_request(body)
