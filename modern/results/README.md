# NeuOS port on Jetson Orin: measured results

Collected with `examples/orin_eval.py` (see `../README.md`), as root, on:

| board | L4T | power mode | baseline (restored after every process) |
|---|---|---|---|
| `orin-nano` Orin Nano 8GB (Super devkit) | R36.5 | 15W (id 0) | cpufreq 729.6..1497.6 MHz (policy0, policy4), cpus 0-5, GPU 306..612 MHz, EMC 2133 MHz `mrq_rate_locked=1` |
| `agx-orin` AGX Orin devkit | R36.4.7 | MAXN (id 0) | cpufreq 729.6..2201.6 MHz (policy0/4/8), cpus 0-11, GPU 306..1300.5 MHz, EMC 3199 MHz `mrq_rate_locked=0` |

Files per board: `baseline.json` (snapshot), `pytest.txt` (on-board `pytest -q`: 53 passed on both),
`{llm,torch}/{configs.txt,profile.csv,Uncertainty.txt,grid.json,profile.log}` and one
`<run>.jsonl` (one line per job) + `<run>.log` per run.

## Grids (config spaces)

Caps: CPU <= baseline `scaling_max_freq`, GPU <= baseline devfreq `max_freq`; EMC from
`possible_rates`, kept only if the rate reads back after the write.

| board / workload | CPU kHz | cores | GPU MHz | EMC kHz | configs | repeats |
|---|---|---|---|---|---|---|
| orin-nano / llm | 115200, 806400, 1497600 | 3, 6 | 306, 408, 510, 612 | 665600, 2133000, 3199000 | 72 | 2 |
| agx-orin / llm | 115200, 1190400, 2201600 | 6, 12 | 306, 612, 1020, 1300.5 | 3199000 (665600, 2133000 dropped: write ignored) | 24 | 3 |
| orin-nano / torch | 115200, 1497600 | 6 | 306, 408, 510, 612 | 665600, 2133000, 3199000 | 24 | 3 |
| agx-orin / torch | 115200, 2201600 | 12 | 306, 612, 1020, 1300.5 | 3199000 | 8 | 3 |

Energy: Nano = `VDD_IN`; AGX = `VDD_GPU_SOC + VDD_CPU_CV + VIN_SYS_5V0 + VDDQ_VDD2_1V8AO`
(idle ~5 W and ~8.5 W; under LLM load ~12 W and 16-40 W).

## Workloads and period

- **llm**: two `llama-server` instances (`-ngl 99 -c 1024 -np 2`): full tier Qwen2.5-1.5B-Instruct
  Q4_K_M on :18081, approx tier Qwen2.5-0.5B Q4_K_M on :18082. Five fixed prompts in rotation,
  `max_tokens 64`, `temperature 0`, `ignore_eos`, `cache_prompt false`, chunk 8 tokens ->
  9 stages (stage 0 = request + first token, the only approximable stage).
- **torch**: torchvision resnet18, random weights, 1x3x224x224, CUDA, `no_grad`; 60 stages
  (leaf-module calls); 17 approximable stages (conv1 7x7 + the 16 3x3 convs) with SVD
  low-rank twins (Kx1 then 1xK) at rank = ceil(0.25 x full rank). With random weights (flat
  spectrum) the all-twins network output has relative error 0.90, so this is a timing /
  policy workload only, not an accuracy one.
- **period** = 1.3 x the median wall latency of warm jobs at the fastest config (last config in
  the sorted table), measured by `profile`: Nano llm 2243 -> 2916 ms, AGX llm 761 -> 990 ms,
  Nano torch 20.0 -> 26 ms, AGX torch 11.5 -> 15 ms. 1.3x leaves room for scheduling overhead
  and one slower config while staying well below the slowest configs (Nano llm up to ~9 s,
  AGX llm 2.6 s). For torch the controller's per-stage overhead (schedule + cohort read, ~2-3
  ms per job, `sched_us`) and Python per-stage hooks make 1.3x infeasible, so one extra run
  uses 2.0x (`*_f2`).

## Summary (30 jobs per run; job 0 = warmup job, excluded)

Latency = wall clock per job (what the deadline is checked against); energy = sum of stage
energies from the controller; approx rate = approximated / approximable stages; configs =
indices into the sorted table (0 = slowest, max = fastest).

| board | run | jobs | period ms | mean ms | p95 ms | met % | energy mJ/job | approx rate | distinct cfgs (min..max) |
|---|---|---|---|---|---|---|---|---|---|
| orin-nano | llm DEFAULT original | 29 | 2916 | 1011.9 | 1010.1 | 100 | 11797 | 1.00 | 2 (49..71) |
| orin-nano | llm DEFAULT corrected | 29 | 2916 | 1011.7 | 1011.4 | 100 | 11832 | 1.00 | 2 (49..71) |
| orin-nano | llm MAX_ACC original | 29 | 2916 | 2246.2 | 2244.6 | 100 | 28489 | 0.00 | 2 (49..71) |
| orin-nano | llm 2 concurrent clients, A | 29 | 2916 | 1202.8 | 1257.9 | 100 | 13903* | 1.00 | 4 (0..71) |
| orin-nano | llm 2 concurrent clients, B | 29 | 2916 | 1201.5 | 1186.2 | 100 | 14041* | 1.00 | 4 (7..71) |
| orin-nano | torch DEFAULT original | 29 | 26 | 31.2 | 36.4 | 0 | 146 | 0.99 | 3 (0..23) |
| orin-nano | torch DEFAULT original, 2.0x period | 29 | 40 | 27.5 | 27.8 | 100 | 136 | 1.00 | 2 (12..23) |
| orin-nano | torch MAX_ACC original | 29 | 26 | 26.2 | 26.1 | 76 | 126 | 0.00 | 2 (12..23) |
| agx-orin | llm DEFAULT original | 29 | 990 | 390.3 | 384.7 | 100 | 10995 | 1.00 | 2 (12..23) |
| agx-orin | llm DEFAULT corrected | 29 | 990 | 389.5 | 388.8 | 100 | 10989 | 1.00 | 2 (12..23) |
| agx-orin | llm MAX_ACC original | 29 | 990 | 763.4 | 766.3 | 100 | 28129 | 0.00 | 2 (12..23) |
| agx-orin | llm 2 concurrent clients, A | 29 | 990 | 478.1 | 476.4 | 100 | 12950* | 1.00 | 2 (4..23) |
| agx-orin | llm 2 concurrent clients, B | 29 | 990 | 470.9 | 475.6 | 100 | 13225* | 1.00 | 2 (1..23) |
| agx-orin | torch DEFAULT original | 29 | 15 | 18.4 | 19.9 | 0 | 165 | 1.00 | 2 (6..7) |
| agx-orin | torch DEFAULT original, 2.0x period | 29 | 23 | 20.8 | 21.2 | 97 | 224 | 1.00 | 2 (6..7) |
| agx-orin | torch MAX_ACC original | 29 | 15 | 20.2 | 20.9 | 14 | 227 | 0.00 | 2 (6..7) |

\* concurrent clients share one board meter, so each client's energy includes the other's
power during its stages (not additive).

Warmup jobs (job 0, random config by seed): Nano llm 2.4-4.6 s, AGX llm 1.1-2.2 s (all miss
the period; they also include llama-server cold paths and CPU hot-plug).

## Observations (behaviour of the faithful port, not fixed)

- **Configs collapse to the max.** In every run, after the warmup job the scheduler moves
  stages 1.. to the fastest config (Nano llm 71, AGX llm 23, torch 23 / 7) and stays there;
  stage 0 is always logged at the warmup's random config because `schedule()` only runs for
  i > 0 (the hardware actually stays at the previous job's last config, since knobs are
  diffed against the table entry). Only the concurrent runs and one Nano torch job show other
  configs (0, 7, 58, 65, 66).
- **Priority saturates.** `priority = mean(1 + slack_us/30)` over the cohort, with slack in
  microseconds, is ~1e4-1e6 in magnitude: positive (torch, corrected variant) or negative
  (original variant's sign quirk lets slack drift to -1e7..-6e7 us after overruns).
  Consequences: DEFAULT approximates *every* approximable stage (rate 1.00), whether through
  priority 1.0 at stage 0 (the job's initial priority; LLM's only approximable stage) or the
  huge positive priority (torch). MAX_ACC never approximates (A* = max_score). So the
  DEFAULT vs MAX_ACC difference is just "always 0.5B / twins" vs "always 1.5B / full".
  No NaN appeared (the empty-cohort guard was never needed after the first stage).
- **original vs corrected** give the same configs and latency here (both saturate); only the
  slack/priority trajectories differ (original: slack monotonically negative; corrected:
  reset each job, stays ~+0.7..1.8e6 us).
- **Cohort sharing works**: with two concurrent clients on one `--cohort-prefix`, the shared
  slot counter interleaves slots (Nano: A 1,3,5,7,9 / B 0,2,4,6,8; single runs: 0..9), and each
  job's priority is computed over the 10 slots written by both processes (e.g. AGX job 10 of
  A reads 10 non-zero slots from both pids). Latency rises ~20% from sharing the GPU.
- **Low-rank twins are not faster on the GPU at batch 1** (2 thin convs instead of 1):
  torch DEFAULT (all twins) vs MAX_ACC: AGX 18.4 vs 20.2 ms, Nano 27.5 (2.0x run) vs 26.2 ms.
- **DVFS actuation time is outside the stage timing** (as in the original): e.g. AGX llm job 1
  wall 661 ms vs stage-sum 20 ms, because onlining cores between stages 0 and 1 takes
  ~0.6 s and the streamed tokens were already buffered when the next stage read them.

## Board-adaptation issues found and fixed

- AGX Orin (MAXN): EMC `clk/emc/rate` writes are accepted but ignored (rate stays 3199 MHz).
  One probe on the AGX (a discarded torch profile) did read back 2133 MHz, so the behaviour is
  not fully consistent; the profiles used here have EMC 3199 only. Nano: all four rates read back.
- devfreq and cpufreq accept writes above the nvpmodel caps (checked on the Nano: GPU 1020 MHz,
  CPU 1728 MHz), so the config space is capped at the baseline maxima.
- Concurrent clients: a cpufreq write raced with the other process hot-plugging the same
  cpus (`EBUSY`) and crashed client B on the first AGX try (logs in
  `agx-orin/llm/failed_first_try/`); per-policy / per-cpu writes now warn and count
  (`dvfs_write_errors` in the JSONL: AGX B 3, Nano B 2) instead of raising.
- `EnergyMeter.mark()` read every rail over I2C synchronously: with 120 marks per resnet18 job
  it made a 11.5 ms job take ~770 ms (first AGX torch profile, discarded). `mark()` now
  extrapolates from the background samples. The two llm profiles were taken with the old
  synchronous mark (9 stages, 30-300 ms each, so the overhead is negligible there); all runs and
  torch profiles use the new one.
- The board's torch 2.8 wheel cannot load CUDA linalg (`cusolverDnXsyevBatched_bufferSize`
  missing), so the SVD for the twins runs on the CPU.
- SIGTERM: `agx-orin/torch/sigterm_test.log` shows a run killed after 370 jobs, mid-stage, with
  the DVFS state pinned at the max config (`sigterm_test.note`), exiting cleanly after
  restoring the exact baseline.
