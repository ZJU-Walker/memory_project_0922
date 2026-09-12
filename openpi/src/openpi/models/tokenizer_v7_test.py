"""v7 phase context: past states and the previous sentence in the FAST-subtask prefix."""

import numpy as np

from openpi.models import tokenizer as _tokenizer


def _tok():
    return _tokenizer.FASTSubtaskTokenizer(max_len=320)


def test_prefix_unchanged_without_context():
    tok = _tok()
    state = np.linspace(-1, 1, 14).astype(np.float32)
    prefix, spans = tok._build_prefix("Make a boba tea", state)  # noqa: SLF001
    assert prefix.startswith("Task: make a boba tea, State: ") and prefix.endswith(";\n")
    assert len(spans) == 1
    start, end = spans[0]
    assert prefix.encode("utf-8")[start:end].decode() == " ".join(map(str, tok._discretize(state)))  # noqa: SLF001
    # The context of tokenize_split is byte-identical to tokenize's prefix (memory path untouched).
    ctx, _ = tok._build_prefix("Make a boba tea", state)  # noqa: SLF001
    assert ctx == prefix


def test_history_and_prev_in_prefix_and_state_mask():
    tok = _tok()
    state = np.zeros(14, np.float32)
    hist = np.stack([np.full(14, -0.5, np.float32), np.full(14, 0.5, np.float32)])
    prefix, spans = tok._build_prefix("make tea", state, hist, "first, scoop, 2 of 3")  # noqa: SLF001
    assert ", Past: " in prefix and " | " in prefix and prefix.endswith(", Last: first, scoop, 2 of 3;\n")
    assert len(spans) == 3
    b = prefix.encode("utf-8")
    assert b[spans[1][0] : spans[1][1]].decode() == " ".join(map(str, tok._discretize(hist[0])))  # noqa: SLF001
    assert b[spans[2][0] : spans[2][1]].decode() == " ".join(map(str, tok._discretize(hist[1])))  # noqa: SLF001
    tokens, mask, ar, loss, fast, state_mask = tok.tokenize(
        "make tea", state, "first, scoop, 3 of 3", None, state_history=hist, prev_subtask="first, scoop, 2 of 3",
        return_state_mask=True,
    )
    n = int(mask.sum())
    assert tokens.shape == (320,) and state_mask.shape == (320,)
    # every digit span is masked, the "Last:" sentence and the subtask segment are not
    pieces = tok._paligemma_tokenizer.encode(prefix, out_type="immutable_proto").pieces  # noqa: SLF001
    for i, p in enumerate(pieces):
        in_span = any(p.begin < e and p.end > s for s, e in spans)
        assert bool(state_mask[i + 1]) == in_span
    assert not state_mask[n - 1]  # the last causal token (subtask newline) is never state
    # more context -> longer prefix than the plain tokenization
    plain = tok.tokenize("make tea", state, "first, scoop, 3 of 3", None)
    assert int(mask.sum()) > int(plain[1].sum())


def test_tokenize_split_matches_plain_context():
    tok = _tokenizer.FASTSubtaskTokenizer(max_len=80)
    state = np.random.default_rng(0).uniform(-1, 1, 14).astype(np.float32)
    ctx, ctx_mask, *_ = tok.tokenize_split("make tea", state, "first, get cup", np.zeros((50, 14), np.float32), 208)
    plain, plain_mask, *_ = tok.tokenize("make tea", state, None, None)
    n = int(ctx_mask.sum())
    assert n == int(plain_mask.sum()) and np.array_equal(ctx[:n], plain[:n])
