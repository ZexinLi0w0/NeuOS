"""Precompute the NeuOS low-rank twins of the SD x4-upscaler UNet once and cache them.

The twins come from neuos_modern.adapters.torch_layers.lowrank_twins (SVD of every KxK conv,
Kx1 + 1xK, rank = ceil(ratio x full rank)); the SVD runs on the CPU (the boards' torch wheel cannot
load CUDA linalg) and takes minutes for the 1024-channel convs, so the result is saved with
torch.save and loaded by the perception task.

    python build_twins.py --board-config configs/orin-nano.json
"""
import argparse
import os
import sys
import time

from common import load_cfg, neuos_path, twins_cache_path

os.environ.setdefault("HF_HUB_OFFLINE", "1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--board-config", required=True)
    a = ap.parse_args()
    cfg = load_cfg(a.board_config)
    neuos_path(cfg)
    import torch
    from diffusers import UNet2DConditionModel
    from neuos_modern.adapters.torch_layers import lowrank_twins
    pc = cfg["perception"]
    out = twins_cache_path(cfg)
    if os.path.exists(out):
        print("exists:", out)
        return
    torch.set_num_threads(os.cpu_count())
    unet = UNet2DConditionModel.from_pretrained(pc["model"], subfolder="unet", variant="fp16",
                                                torch_dtype=torch.float16).eval()
    t0 = time.time()
    twins = lowrank_twins(unet, pc["twin_rank_ratio"])
    print("twins: %d convs in %.1f s" % (len(twins), time.time() - t0), flush=True)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save(twins, out + ".tmp")
    os.replace(out + ".tmp", out)
    print("saved", out)


if __name__ == "__main__":
    sys.exit(main())
