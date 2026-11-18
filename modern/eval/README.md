# NeuOS baseline on the Simmer voice-assistant workload

Harness that runs the NeuOS (ATC'20) baseline, through the faithful Python port
`neuos_modern` (`../neuos_modern` in this fork), on the Simmer paper's voice-assistant
workload on the Jetson Orin Nano 8GB (15W) and the AGX Orin (MAXN). `--policy default` runs
the same workload without NeuOS (top tiers, stock governors, no DVFS writes) so the harness can
host the other baselines too.

It lives in `modern/eval/` and is self-contained: it imports only `neuos_modern` (the board
config's `neuos_modern_dir` when that holds the package, else this fork's `modern/`) and
third-party packages (torch, transformers, diffusers, requests, numpy). It was developed in the
AudiOS repo (branch `neuos-baseline`, 2089208); references to AudiOS scripts below are
provenance only.

## Files

| file | role |
|---|---|
| `harness.py` | launcher (root): `verify`, `profile`, `run`; memory preflight + watchdog, teardown, DVFS restore, verification |
| `tasks.py` | one process per task: `asr`, `perception`, `turn` (LLM + TTS); NeuOS controller per process |
| `sampler.py` | 1 Hz board sampler + 20 Hz energy integral |
| `listen_trace.py` | seeded Listen/Respond trace generator |
| `summarize.py` | paper metrics per run (definitions in its docstring) |
| `build_twins.py` | precomputes the UNet SVD low-rank twins once (CPU SVD takes 2.5 min on the AGX, 5.6 min on the Nano) |
| `common.py` | shared helpers |
| `fetch_clips.py` | builds the LibriSpeech clip file (`clips_npz`) the ASR task reads (vendored from the AudiOS rtaudio cache code) |
| `configs/{orin-nano,agx-orin}.json` | every per-board choice (tiers, periods, deadlines, grid, memory floor, baseline) |
| `push.sh`, `pull.sh` | upload the harness / fetch a run directory (via `/tmp/jssh2.sh`, `/tmp/jraw.sh`) |
| `results/smoke/<board>/<run>/` | 3-minute smoke-test logs pulled from the boards, plus the profiles (`profiles/`) |
| `results/full/<board>/<policy>/` | 20-minute runs; the large `perception.jsonl` / `samples.jsonl` are gzipped (`summarize.py` reads `.jsonl.gz`) |

## How NeuOS is mapped onto the workload

- Every task is its own process with its own `NeuOSController`; all share one
  `SharedCohort("neuos_eval")`, the same config space (`profiles/<board>/configs.txt`) and the
  saved baseline (`OrinSysfsBackend(baseline=...)`), as the original concurrent Caffe processes.
- Approximation = NeuOS's binary full-vs-twin choice. Full = the highest tier of the task (what
  Default runs), approx = the next lower tier; both are resident.
  - ASR (Whisper): stage 0 = tier decision + features + encoder; then greedy decoder chunks of
    8 tokens (max 32). Period = deadline = 1000 ms, releases only in Listen phases.
  - Perception: SD x4-upscaler UNet, one denoising step per frame on a 32x32 latent
    (as `frame_loop.py` in the AudiOS repo's `motivation/scripts`); per-layer NeuOS (`torch_layers` adapter, 531
    stages) with SVD low-rank twins (rank 0.25) of the 52 KxK convs. Period = deadline = 200 ms.
  - Turn: stage 0 = tier decision + request + first token (llama-server `/v1/completions`
    stream to the full or approx server); then 8-token chunks (max 64); as soon as the text
    holds a complete first sentence a TTS stage (MMS-TTS) synthesises it (= first audio); a last
    stage synthesises the rest. TTS is never approximated. NeuOS period = turn deadline.
- Release model: ASR / perception are periodic on a grid anchored at the start of the measured
  window; a release that occurs while the previous job runs is dropped (counted as a miss).
- Trace: `listen_trace.py` (seed 0): Listen 5-15 s (ASR on the LibriSpeech clips, 73
  validation-clean utterances, `fetch_clips.py`), then one turn released at the end of
  Listen (fixed prompt list), then 2-6 s idle. Perception runs throughout.
- NeuOS warmup: after all models are loaded and warmed up, every controller runs its first
  (warmup) job concurrently, outside the measured window.
- Profile (Uncertainty): every task alone with its full tier over the whole config space
  (`neuos_modern.profile`), 2 repeats. Config space: CPU {729.6 MHz, max} with all cores online,
  GPU 3 levels up to the power-mode cap, EMC rates that read back after a write (Nano: 665.6 /
  2133 MHz; 3199 MHz is left out because 15W locks EMC at 2133. AGX: 665.6 / 2133 / 3199 MHz in
  the canonical profile; the probe is not consistent on the AGX, an earlier core-mask profile
  kept 3199 only). Nano 12 configs, AGX 18.

## Deliberate deviation from original NeuOS: no CPU hot-plugging

The original NeuOS configurations include a CPU core mask, and the port supports it. Here the
config space has **all cores online** (`grid.cores = ["all"]` in both board configs); every other
NeuOS knob (CPU frequency, GPU frequency, EMC rate) is kept. Reason: with cores {all, half},
per-layer NeuOS on the UNet switched ~7 times per frame between a half-core config and the
fastest config, so every frame took CPUs offline and online again: ~3.3 s of DVFS actuation per
frame on the AGX (`dvfs.apply_ms` in `perception.jsonl`), every frame late or dropped, and the
hot-plugs also delayed ASR (0.3-0.9 s per job). The core-mask runs are kept for reference as
`results/smoke/<board>/neuos_coremask*` (profiles `profiles/<board>-coremask`). To re-enable
hot-plug: `--set 'grid.cores=["all","half"]'` with a separate `--profile-dir`.

## Smoke-test findings (3 min per run, `results/smoke/`)

- Core masks: see the deviation above (`neuos_coremask*` runs). `agx-orin/neuos_allcores_prelim`
  was the first all-cores trial (its own profile, same 18 configs as the canonical one).
- Even without DVFS, per-layer hooks (531 stages, a CUDA sync each) make a frame 120 ms (AGX)
  and 228 ms (Nano) alone at the fastest config, so per-layer perception cannot meet 200 ms on
  the Nano; low-rank twins are slower than the full convs at batch 1 (as in the NeuOS port's
  results).
- DEFAULT mode always approximates (priority saturates, as documented in the port): ASR always
  runs the lower tier, the LLM always the approx server, the UNet always the twins.

## Per-board decisions

| | Nano 8GB 15W | AGX Orin MAXN |
|---|---|---|
| ASR full / approx | whisper-base / whisper-tiny (reduced, memory) | whisper-small / whisper-base |
| LLM full / approx | Qwen2.5-0.5B Q4_K_M / none (reduced, memory) | Qwen2.5-7B / 3B Q4_K_M |
| perception | UNet + low-rank twins | UNet + low-rank twins |
| periods / deadlines | ASR 1000, perception 200, turn 2100 ms (smoke: 4000) | ASR 1000, perception 200, turn 2000 ms |
| memory floor | 600 MB (800 MB until the floor800 run) | 2000 MB |

Turn deadline = about 1.3x the Default turn latency p50 measured in the smoke runs (AGX 1546 ms
-> 2000 ms; Nano 1604 ms -> 2100 ms; the Nano smoke runs used 4000 ms).

Nano: the requested pair (3B / 1.5B, whisper-small / base) does not fit. Measured
MemAvailable cost per component: perception 1.9 GB (2.1 with twins), ASR (small) 1.1-1.5 GB,
turn (TTS) ~1.0 GB, llama-server 1.5B ~1.0 GB, 0.5B ~0.2-0.6 GB; MemAvailable at idle ~5.8 GB.
Even Default with 1.5B tripped the 800 MB watchdog while loading the turn process (twice,
aborted cleanly: `results/smoke/orin-nano/default_try*_memabort`). With the reduced tiers the
minimum MemAvailable was 1201 MB (Default) and 821 MB (NeuOS, 21 MB above the floor), so a
20-minute NeuOS run on the Nano may still end in a clean watchdog abort.

**Nano floor lowered to 600 MB.** NeuOS keeps both tiers of every task resident (whisper
base + tiny, the UNet + its low-rank twins), so it needs more memory than Default. At the 800 MB
floor the 20-minute Nano NeuOS run was stopped cleanly by the watchdog after ~4 min (min 799 MB,
kept as `results/full/orin-nano/neuos_floor800`). 600 MB matches the earlier
AudiOS `motivation/scripts/asr_loop.py` default (`FLOOR_MB=600`). Default was not rerun: its minimum
over 20 min was 1173 MB, so the floor never mattered there. `mem_margin_mb` stays 150: a lower
floor only loosens the preflight, so nothing that loaded before is refused now. The 20-minute Nano
NeuOS rerun at 600 MB (`results/full/orin-nano/neuos`) completed without an abort; its minimum
MemAvailable was 831 MB and dmesg showed no OOM-killer lines.

## Smoke results (3 min)

Miss ratios are ASR / perception / turn / all jobs; turn = end of Listen to first audio.

| board | run (dir) | config space | miss A / P / T / all | turn p50 / p99 ms | tok/s | TTS RTF p50 / p99 | min avail MB | throttled % | peak Tj C | energy J (W) |
|---|---|---|---|---|---|---|---|---|---|---|
| AGX | default (`default`) | - | .049 / .002 / .182 / .010 | 1546 / 2250 | 18.6 | .035 / .093 | 17126 | 2.2 | 53.5 | 3562 (19.9) |
| AGX | **neuos (`neuos`)** | all cores, 18 cfgs | .000 / .301 / .000 / .261 | 686 / 984 | 40.2 | .029 / .081 | 14294 | 0 | 54.4 | 3437 (19.2) |
| AGX | neuos (`neuos_coremask`) | cores {12, 6}, 12 cfgs | .355 / 1.000 / .000 / .912 | 899 / 1284 | 34.6 | .043 / .128 | 14432 | 0 | 48.9 | 2239 (12.5) |
| Nano | default (`default`) | - | .026 / .302 / .400 / .272 | 1519 / 2639 | 28.0 | .084 / .217 | 1189 | 0 | 53.7 | 1557 (8.7) |
| Nano | **neuos (`neuos`)** | all cores, 12 cfgs | .000 / 1.000 / .000 / .874 | 1288 / 1976 | 34.9 | .075 / .155 | 854 | 0 | 51.8 | 1312 (7.3) |
| Nano | neuos (`neuos_coremask`) | cores {6, 3}, 24 cfgs | .000 / 1.000 / .000 / .874 | 1130 / 1993 | 35.5 | .073 / .149 | 821 | 0 | 51.8 | 1312 (7.3) |

No OOM events. Nano turn deadline: 4000 ms in `neuos_coremask` and `default_ub512`, 2100 ms in
`default` and `neuos`. The Nano `neuos` and `default` runs use `llm.server_args = -b 128 -ub 128`
(smaller llama-server logits / compute buffer, ~75 MB): the first all-cores Nano NeuOS run
(`neuos_try1_memabort`, default 512 batch) was stopped cleanly by the watchdog 15 s into the
window at 799 MB; `default_ub512` is the Default run with the 512 batch. Other kept runs:
`agx-orin/neuos_coremask_try1` (same as `neuos_coremask`, without DVFS instrumentation),
`agx-orin/neuos_allcores_prelim`, `orin-nano/default_try{1,2}_memabort` (Default with whisper-small
and Qwen 1.5B, watchdog stop while loading).

## 20-minute results (`results/full/`)

`python summarize.py results/full/*/*` (from `modern/eval`):

| board | policy | miss ASR / perc / turn / all | turn p50 / p99 ms | tok/s | TTS RTF p50 / p99 | OOM | min avail MB | throttled % | peak Tj C | energy J (W) | abort |
|---|---|---|---|---|---|---|---|---|---|---|---|
| AGX | default | .028 / .000 / .221 / .006 | 1600 / 2427 | 19.3 | .035 / .091 | 0 | 16851 | 0.0 | 56.2 | 24090 (20.1) | - |
| AGX | neuos | .000 / .280 / .000 / .244 | 677 / 1019 | 40.1 | .028 / .106 | 0 | 14197 | 0.0 | 55.9 | 23094 (19.3) | - |
| Nano | default | .024 / .298 / .162 / .267 | 1438 / 2618 | 29.2 | .087 / .214 | 0 | 1173 | 0.0 | 54.2 | 10347 (8.6) | - |
| Nano | neuos (floor 600 MB) | .000 / 1.000 / .000 / .878 | 1103 / 1984 | 36.5 | .070 / .149 | 0 | 831 | 0.0 | 52.7 | 8761 (7.3) | - |
| Nano | neuos_floor800 | .000 / 1.000 / .000 / .873 | 1164 / 1979 | 36.0 | .074 / .151 | 0 | 799 | 0.0 | 51.8 | 1822 (7.3) | stopped at ~4 min |

## TTS on the CPU (`tts.device`, tried on the Nano, not adopted)

`tts.device` = `"cuda"` (default) or `"cpu"`. With `"cpu"` the turn process never touches CUDA
(no CUDA context; `release_host_memory()` checks `torch.cuda.is_initialized()` first and the
`loaded` record logs `cuda_context: false`). Measured on the Nano (NeuOS, 3 min,
`results/smoke/orin-nano/neuos_ttscpu`): loading the turn process cost 221 MB of MemAvailable
instead of ~1.55 GB, MemAvailable at the start of the window was 1674 MB instead of 930 MB
(+744 MB), and the minimum over the run was 1256 MB instead of 854 MB (+402 MB). But MMS-TTS
(VITS, fp32) on the Nano's six A78 cores is slower than real time: RTF p50 1.25, p99 1.79 under
NeuOS (the CPU at 1497.6 MHz most of the time), and the turn profile alone at the fastest config
took 19.5 s per 64-token turn vs 2.1 s with TTS on the GPU (`profiles/orin-nano-ttscpu`). Every
turn missed (turn latency p50 10.0 s). The Nano config therefore keeps `"cuda"`; the canonical
Nano turn profile on the board is the GPU-TTS one (the CPU-TTS profile is in
`profiles/orin-nano/ttscpu/`).

## Commands

Upload from the fork (B=agx-orin P=8002, or B=orin-nano P=8010): the harness goes to
`/experiment/zexin/neuos_eval/harness`, the package to `/experiment/zexin/neuos/modern`
(`neuos_modern_dir` in the board configs).

```sh
cd modern/eval; export GW=<gateway-ip> B=agx-orin P=8002
./push.sh $P                                                       # harness (results excluded)
git -C ../.. archive HEAD modern | /tmp/jssh2.sh $P 'mkdir -p /experiment/zexin/neuos && cd /experiment/zexin/neuos && tar xmf -'   # -> neuos/modern
```

The canonical all-cores profiles are on the boards in
`/experiment/zexin/neuos_eval/profiles/{agx-orin,orin-nano}`. One run per board at a time;
start the next after the log ends with `VERIFY OK`:

```sh
/tmp/jssh2.sh $P "cd /experiment/zexin/neuos_eval/harness && (sudo -n setsid nohup /experiment/zexin/venvs/r3/bin/python harness.py run --board-config configs/$B.json --policy neuos --minutes 20 --profile-dir /experiment/zexin/neuos_eval/profiles/$B --out-dir /experiment/zexin/neuos_eval/runs/full/$B/neuos > ../full_${B}_neuos.log 2>&1 < /dev/null &)"
/tmp/jssh2.sh $P "cd /experiment/zexin/neuos_eval/harness && (sudo -n setsid nohup /experiment/zexin/venvs/r3/bin/python harness.py run --board-config configs/$B.json --policy default --minutes 20 --out-dir /experiment/zexin/neuos_eval/runs/full/$B/default > ../full_${B}_default.log 2>&1 < /dev/null &)"
./pull.sh $P /experiment/zexin/neuos_eval/runs/full/$B/neuos results/full/$B/neuos
python3 summarize.py results/full/*/*
```

On the board (setup / re-profiling):

```sh
cd /experiment/zexin/neuos_eval/harness; PY=/experiment/zexin/venvs/r3/bin/python
sudo -n $PY harness.py verify --board-config configs/agx-orin.json
HF_HOME=/experiment/zexin/hf $PY build_twins.py --board-config configs/agx-orin.json      # once
$PY fetch_clips.py --out /experiment/zexin/neuos_eval/data/librispeech_dummy.npz           # once, needs network
sudo -n $PY harness.py profile --board-config configs/agx-orin.json --profile-dir ../profiles/agx-orin
```

`--set KEY=JSON` overrides a config entry (with a separate `--profile-dir` when it changes the
grid); `--turn-deadline-ms` overrides the turn deadline of a run.

Safety: the launcher refuses to start unless the board matches `expected_baseline` and no
harness process is left; every task process restores the saved DVFS snapshot in `finally` and
on SIGTERM / SIGINT; the launcher restores it again after all tasks exit (neuos), unlinks the
cohort blocks, and prints the verification. A 4 Hz watchdog (and a check before every job)
stops the run cleanly if MemAvailable drops below the floor; a preflight refuses to load a
component whose estimated footprint would cross floor + margin.
