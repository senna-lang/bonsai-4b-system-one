"""Weight-free checks of request validation, prompt packing, and position IDs."""
import pytest

from system_one_bonsai import Branch, PackedRequest, pack_request, position_ids


class CharTokenizer:
    """Stand-in for a chat template: one token per character after a fixed header."""

    def apply_chat_template(self, messages, **_):
        return [0, 0] + [ord(c) for c in messages[0]["content"]] + [1]


def request(*branches):
    return PackedRequest(state="A parcel arrived late and damaged.", branches=list(branches))


def test_branches_share_the_state_prefix_and_keep_their_own_suffix():
    tok = CharTokenizer()
    branches = [
        Branch("choice", "Which problem?", ["Late", "Damaged", "Lost"]),
        Branch("noul", "Was it damaged?"),
    ]
    ids, state_len, spans = pack_request(request(*branches), tok)
    for branch, (start, end) in zip(branches, spans):
        alone, _, ((s, e),) = pack_request(request(branch), tok)
        # Prefix + this branch's suffix reconstructs the stand-alone prompt exactly.
        assert ids[:state_len] + ids[start:end + 1] == tok.apply_chat_template(
            [{"role": "user", "content": "".join(chr(c) for c in alone[2:-1])}]
        )
    assert spans[0][0] == state_len and spans[1][0] == spans[0][1] + 1 and spans[1][1] == len(ids) - 1


def test_single_branch_keeps_at_least_one_suffix_token():
    ids, state_len, ((start, end),) = pack_request(request(Branch("noul", "Late?")), CharTokenizer())
    assert start == state_len < len(ids) and end == len(ids) - 1


def test_each_branch_restarts_positions_after_the_shared_prefix():
    assert position_ids(3, [(3, 5), (6, 7)]) == [0, 1, 2, 3, 4, 5, 3, 4]


def test_branch_order_does_not_change_a_branch_suffix():
    tok = CharTokenizer()
    a, b = Branch("noul", "Late?"), Branch("score", "How bad?", ["Low", "High"])
    ids1, n1, s1 = pack_request(request(a, b), tok)
    ids2, n2, s2 = pack_request(request(b, a), tok)
    assert ids1[s1[0][0]:s1[0][1] + 1] == ids2[s2[1][0]:s2[1][1] + 1]


@pytest.mark.parametrize(
    "kind, options",
    [("choice", ["Only"]), ("score", [str(i) for i in range(21)]), ("noul", ["Yes", "No"]), ("other", ["a", "b"])],
)
def test_invalid_branches_are_rejected(kind, options):
    with pytest.raises(ValueError):
        Branch(kind, "q", options)


def test_empty_request_is_rejected():
    with pytest.raises(ValueError):
        pack_request(PackedRequest(state="s", branches=[]), CharTokenizer())
