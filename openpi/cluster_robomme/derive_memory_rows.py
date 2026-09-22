"""Which sentence tokens NEED the memory? A label-driven rule (09-15, user: digit blinding "is too task specific").

For every memory step the model predicts the current sentence token by token. Walk every training episode's label
sequence (sidecar segments, consecutive duplicates collapsed) and, for each bank state H = the sentences already written
(seq[:j]), the two sentences the model may have to produce are seq[j-1] (still inside the previous segment) and seq[j]
(its onset). For each token position of those sentences record the set of possible next tokens under two keys:
  * (prompt, prefix)           -- what the text alone already fixes;
  * (prompt, history, prefix)  -- what the text plus the bank fixes.
A trie node (prompt-independent prefix) is
  * SIGHTED/TEXT  if the prompt + prefix always fix the next token (colour, filler words) -> no memory needed;
  * MEMORY        if the prompt + prefix do NOT fix it but the history always does (the ordinal) -> blind row;
  * TIMING        if even the history leaves >1 option (the phase word: same sentence or the next one) -> needs vision.
Prints the MEMORY nodes as suffix patterns for `memory_v7_digit_blind_patterns` (full prefix ids) and the TIMING nodes.
"""
import argparse
import collections
import json
import pathlib

import sentencepiece


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sidecar", type=pathlib.Path, required=True)
    ap.add_argument("--manifest", type=pathlib.Path, required=True)
    ap.add_argument("--tokenizer", type=pathlib.Path,
                    default=pathlib.Path(__file__).resolve().parents[2] / "v35/cache/openpi/big_vision/paligemma_tokenizer.model")
    ap.add_argument("--prompt-context", action="store_true", default=True)
    args = ap.parse_args()
    sp = sentencepiece.SentencePieceProcessor(model_file=str(args.tokenizer))
    sidecar = json.loads(args.sidecar.read_text())
    manifest = json.loads(args.manifest.read_text())
    prompts = {e["stable_id"]: (e.get("task_goals") or [e.get("prompt", "")])[0] for e in manifest["episodes"]}
    enc = {}
    def tokens(s):
        if s not in enc:
            enc[s] = tuple(sp.encode(s.lower().strip()) + sp.encode("\n"))
        return enc[s]
    by_prompt = collections.defaultdict(set)   # (prompt, prefix) -> next tokens
    by_hist = collections.defaultdict(set)     # (prompt, history, prefix) -> next tokens
    nodes = {}                                 # prefix -> pieces
    for sid, ep in sidecar["episodes"].items():
        prompt = prompts[sid]
        seq = []
        for seg in ep["segments"]:
            if not seq or seq[-1] != seg["sentence"]:
                seq.append(seg["sentence"])
        for j in range(len(seq)):
            hist = tuple(seq[:j])
            candidates = {seq[j]} | ({seq[j - 1]} if j else set())
            for sent in candidates:
                row = tokens(sent)
                for i in range(len(row)):
                    prefix = row[:i]
                    nodes.setdefault(prefix, tuple(sp.id_to_piece(t) for t in prefix))
                    by_prompt[(prompt, prefix)].add(row[i])
                    by_hist[(prompt, hist, prefix)].add(row[i])
    prompt_fixed = collections.defaultdict(lambda: True)
    hist_fixed = collections.defaultdict(lambda: True)
    alternatives = collections.defaultdict(set)
    for (prompt, prefix), nxt in by_prompt.items():
        prompt_fixed[prefix] &= len(nxt) == 1
        alternatives[prefix] |= nxt
    for (prompt, hist, prefix), nxt in by_hist.items():
        hist_fixed[prefix] &= len(nxt) == 1
    memory, timing = [], []
    for prefix, pieces in sorted(nodes.items()):
        if prompt_fixed[prefix]:
            continue
        alts = sorted(sp.id_to_piece(t) for t in alternatives[prefix])
        (memory if hist_fixed[prefix] else timing).append((prefix, pieces, alts))
    print(f"{args.sidecar.name}: {len(sidecar['episodes'])} episodes, {len(sidecar.get('sentences', []))} sentences, "
          f"{len(nodes)} trie nodes; {len(memory)} MEMORY rows, {len(timing)} TIMING rows")
    for prefix, pieces, alts in memory:
        print(f"  MEMORY  prefix={' '.join(pieces) or '<start>'!r:60} next in {alts}  pattern={list(prefix)}")
    for prefix, pieces, alts in timing:
        print(f"  TIMING  prefix={' '.join(pieces) or '<start>'!r:60} next in {alts}")


if __name__ == "__main__":
    main()
