"""One real batch of pi05_robomme_0920_v0_smoke through the training loader on the node (CPU JAX): shapes, history masks,
prompt lengths, image-key order. No model, no GPU."""
import os, sys, time
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax, numpy as np
from openpi.training import config as _config
from openpi.training import data_loader as _dl
def main():
    t0 = time.time()
    cfg = _config.get_config(sys.argv[1] if len(sys.argv) > 1 else "pi05_robomme_0920_v0_smoke")  # argv[1]: config name
    cfg = cfg.__class__(**{**cfg.__dict__, "num_workers": 8}) if False else cfg
    loader = _dl.create_data_loader(cfg, sharding=None, shuffle=True, num_batches=2, exact_resume=False)
    print(f"loader built in {time.time()-t0:.0f}s", flush=True)
    for bi, (obs, actions) in enumerate(loader):
        print(f"--- batch {bi} ({time.time()-t0:.0f}s)")
        print("images:", {k: (tuple(v.shape), str(v.dtype)) for k, v in obs.images.items()})
        print("image_masks:", {k: tuple(v.shape) for k, v in obs.image_masks.items()})
        step = np.asarray(obs.seq_step_mask)
        print("state", obs.state.shape, "prompt", obs.tokenized_prompt.shape, "causal", obs.tokenized_causal.shape, "actions", actions.shape,
              "| valid steps per sample", step.sum(-1).tolist())
        for k in sorted(obs.image_masks):
            m = np.asarray(obs.image_masks[k])
            print(f"  mask {k}: valid {m.mean():.3f} | first-step valid per sample {m[:, 0].astype(int).tolist()}")
        hist = [k for k in obs.images if k.startswith("history_")]
        if hist:
            # a padded history frame (mask False) repeats frame 0 = the current frame of step 0
            img0 = np.asarray(obs.images["base_0_rgb"][:, 0]); h0 = np.asarray(obs.images[hist[0]][:, 0]); m0 = np.asarray(obs.image_masks[hist[0]][:, 0])
            same = np.array([np.allclose(img0[i], h0[i]) for i in range(img0.shape[0])])
            print("  step-0 oldest-history == current frame per sample:", same.astype(int).tolist(), "| its mask:", m0.astype(int).tolist())
            # at a later valid step the history differs from the current frame (motion)
            ks = 5
            if step.shape[1] > ks:
                imgk = np.asarray(obs.images["base_0_rgb"][:, ks]); hk = np.asarray(obs.images[hist[-1]][:, ks])
                diff = [float(np.abs(imgk[i] - hk[i]).mean()) for i in range(imgk.shape[0])]
                print(f"  step-{ks} mean |newest history - current| per sample:", [round(d, 4) for d in diff])
        pm = np.asarray(obs.tokenized_prompt_mask)
        print("  prompt tokens used max", int(pm.sum(-1).max()), "of", pm.shape[-1], "| slot fields:", obs.token_slot_mask is not None,
              "| label_write_prob field:", obs.seq_label_write_prob)
    print("done", flush=True)


if __name__ == "__main__":
    main()
