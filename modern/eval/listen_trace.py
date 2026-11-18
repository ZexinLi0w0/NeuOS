"""Seeded Listen/Respond trace generator.

A trace is a list of phases: Listen for ``listen_s`` (uniform in [5, 15] s; ASR active, the user
speaks LibriSpeech clip ``clip`` onward), then Respond: one turn request (prompt ``prompt``)
released at the end of Listen, lasting until the turn finishes, then ``idle_s`` (uniform in
[2, 6] s) of silence. Respond's length depends on the system, so the run length is set by the
wall clock (``--minutes``) and the trace holds enough phases to cover it.

    python listen_trace.py --minutes 20 --seed 0 --out trace.json
"""
import argparse
import json
import random


def make_trace(minutes, seed=0, listen_s=(5, 15), idle_s=(2, 6), n_prompts=12, n_clips=73):
    rng = random.Random(seed)
    n = int(minutes * 60 / (listen_s[0] + idle_s[0])) + 2      # upper bound on phases needed
    phases = [{"listen_s": round(rng.uniform(*listen_s), 2), "idle_s": round(rng.uniform(*idle_s), 2),
               "prompt": rng.randrange(n_prompts), "clip": rng.randrange(n_clips)} for _ in range(n)]
    return {"seed": seed, "minutes": minutes, "listen_s": list(listen_s), "idle_s": list(idle_s),
            "phases": phases}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    with open(a.out, "w") as f:
        json.dump(make_trace(a.minutes, a.seed), f, indent=1)


if __name__ == "__main__":
    main()
