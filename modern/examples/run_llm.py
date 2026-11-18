"""Run NeuOS over a streaming LLM server (llama-server / vllm serve), one job per prompt.

Example (Orin, as root):
    python examples/run_llm.py --configs orin.config --uncertainty Uncertainty.txt \\
        --period-ms 4000 --ladder examples/ladders.json --prompts prompts.txt --jobs 50
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neuos_modern import (DryRunBackend, NeuOSController, NullEnergyMeter, OrinSysfsBackend,  # noqa: E402
                          SharedCohort, build_config_space, load_configs, load_ladders,
                          load_uncertainty)
from neuos_modern.adapters.openai_stream import stream_completion  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--configs", help="config file (ID CPU_freq mask... GPU_freq mem_freq)")
    src.add_argument("--enumerate", action="store_true", help="build the config space from sysfs")
    ap.add_argument("--stride", type=int, nargs=3, default=(1, 1, 1), metavar=("CPU", "GPU", "EMC"),
                    help="frequency subsampling strides for --enumerate")
    ap.add_argument("--uncertainty", required=True)
    ap.add_argument("--period-ms", type=float, required=True)
    ap.add_argument("--mode", default="DEFAULT", choices=("DEFAULT", "MAX_ACC", "MIN_ENG"))
    ap.add_argument("--variant", default="original", choices=("original", "corrected"))
    ap.add_argument("--dry-run", action="store_true", help="record DVFS writes instead of applying them")
    ap.add_argument("--ladder", default=str(Path(__file__).with_name("ladders.json")))
    ap.add_argument("--ladder-name", default="llm")
    ap.add_argument("--prompts", required=True, help="one prompt per line")
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--chunk-tokens", type=int, default=8)
    ap.add_argument("--cohort-prefix", default="neuos")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--out", default="neuos_llm.jsonl")
    a = ap.parse_args()

    backend = DryRunBackend() if a.dry_run else OrinSysfsBackend()
    if a.enumerate:
        s = OrinSysfsBackend().enumerate_states()
        configs = build_config_space(s["cpu_freqs_khz"], s["gpu_freqs_hz"], s["emc_freqs_khz"],
                                     range(1, s["n_cpus"] + 1), s["n_cpus"], *a.stride)
    else:
        configs = load_configs(a.configs)
    prompts = [p for p in Path(a.prompts).read_text().splitlines() if p.strip()]
    ladder = load_ladders(a.ladder)[a.ladder_name]

    energy = NullEnergyMeter()
    if not a.dry_run:
        from neuos_modern.energy import EnergyMeter, find_rails
        if find_rails():
            energy = EnergyMeter()
    cohort = SharedCohort(a.cohort_prefix)
    ctl = NeuOSController(configs, load_uncertainty(a.uncertainty), a.period_ms, backend, cohort,
                          mode=a.mode, variant=a.variant, seed=a.seed, energy=energy)
    energy.start()
    try:
        with open(a.out, "a") as out:
            for j in range(a.jobs):
                text, log = stream_completion(ctl, ladder, prompts[j % len(prompts)],
                                              a.max_tokens, a.chunk_tokens)
                log.update(job=j, mode=a.mode, variant=a.variant)
                out.write(json.dumps(log) + "\n")
                out.flush()
                print("job %d: %.1f ms, %.1f mJ, tier=%s" % (j, log["latency_us"] / 1e3,
                                                              log["energy_mj"], log["tier"]))
    finally:
        energy.stop()
        backend.restore()
        cohort.close()


if __name__ == "__main__":
    main()
