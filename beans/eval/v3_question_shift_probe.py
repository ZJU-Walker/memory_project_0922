"""How big is v3's 'last note' question shift compared with the 8 base questions? (CPU, no data)
For each sentence in the vocabulary used as the last committed note: norm of Z_prev(note) vs the norms of the 8 base
questions P_q(base_i), and the cosine between the 8 shifted questions (1.0 = all eight ask the same thing)."""
import argparse, json, pathlib
import numpy as np, jax.numpy as jnp
import sentencepiece
import openpi.models.model as _model, openpi.training.config as _config, openpi.shared.project_paths as project_paths
from openpi.models import memory as _memory
ap = argparse.ArgumentParser(); ap.add_argument("--config-name", required=True); ap.add_argument("--params", type=pathlib.Path, required=True)
ap.add_argument("--output", type=pathlib.Path, required=True); args = ap.parse_args()
cfg = _config.get_config(args.config_name); model = cfg.model.load(_model.restore_params(args.params, restore_type=np.ndarray)); model.eval()
L = cfg.model.memory_v5_sentence_len
sp = sentencepiece.SentencePieceProcessor(model_file=str(project_paths.project_path('v35/cache/openpi/big_vision/paligemma_tokenizer.model')))
base = jnp.asarray(np.asarray(model.memory_sem_read_query_bank.value, np.float32))
pre = np.asarray(model.memory_sem_query_proj(base).astype(jnp.float32)); pn = np.linalg.norm(pre, axis=-1)
def unit(a): return a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-12)
def pair_cos(Q): U = unit(Q); C = U @ U.T; return float((C.sum() - len(Q)) / (len(Q) * (len(Q) - 1)))
out = {"base_question_norms": pn.tolist(), "mean_cos_between_base_questions": pair_cos(pre), "notes": {}}
print(f"base questions: norms {np.round(pn, 3).tolist()} | mean cos between the 8 base questions {out['mean_cos_between_base_questions']:.3f}", flush=True)
for row in cfg.model.memory_v5_reference_tokens:
    row = [int(t) for t in row]; tok = np.zeros((1, L), np.int32); msk = np.zeros((1, L), bool); tok[0, :len(row)] = row; msk[0, :len(row)] = True
    emb = np.asarray(model.PaliGemma.llm(jnp.asarray(tok), method="embed").astype(jnp.float32))
    note = (emb * msk[..., None]).sum(1) / max(msk.sum(), 1)
    shift = np.asarray(model.memory_sem_query_prev_proj(jnp.asarray(note)).astype(jnp.float32))[0]
    shifted = pre + shift[None]
    text = sp.decode([t for t in row if t not in (108, 1)]).strip()
    rec = {"shift_norm": float(np.linalg.norm(shift)), "mean_cos_between_shifted_questions": pair_cos(shifted),
           "cos_shifted_vs_base_per_question": [float(unit(shifted[i]) @ unit(pre[i])) for i in range(len(pre))]}
    out["notes"][text] = rec
    print(f"note {text[:44]:44s}: shift norm {rec['shift_norm']:.3f} (base {pn.mean():.3f}) | cos between the 8 shifted questions {rec['mean_cos_between_shifted_questions']:.3f} | shifted vs own base {np.round(rec['cos_shifted_vs_base_per_question'], 2).tolist()}", flush=True)
args.output.write_text(json.dumps(out, indent=1)); print("wrote", args.output)
