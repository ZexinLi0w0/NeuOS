# neuos-modern

A faithful Python port of the runtime coordination policy of **NeuOS** (USENIX ATC'20):
history-based DVFS scheduling (`include/triangle/schedule.h`), the per-layer
slack / cohort / approximation loop of `Net::ForwardFromTo` (`src/caffe/net.cpp`), and the
offline Uncertainty profiler (`hashtables/generate_hashtable.py`). It is meant as a baseline:
by default it reproduces the original behaviour, quirks included. The original Caffe code in
this repository is left untouched.

The core is stdlib-only (Python >= 3.9). `torch` and `requests` are imported lazily, only by the
adapters that need them.

## Mapping

| Original (Caffe, TX2 / AGX Xavier) | Port |
|---|---|
| Caffe layer, `ForwardFromTo` loop | stage; `job.before_stage(i, approximable)` / `job.after_stage(i)` (`controller.py`) |
| `Net::setup_DVFS`, `myCompare` | `NeuOSController(...)`, `config.sort_configs` (mem, CPU, GPU freq, #cores) |
| `Net::warmup` | first job: random config (`seed`), deadlines = measured stage latencies |
| `schedule()` overloads, `hist_ent` | `schedule.schedule`, `schedule.schedule_basic`, `HistEnt` |
| low-rank twin of a Convolution layer (`models/lowrank`) | next lower model tier (`tiers.TierLadder`), or `torch_layers.lowrank_twins` (SVD Kx1 + 1xK) |
| TX2 / AGX `setconfig.h` sysfs | `dvfs.OrinSysfsBackend` (Orin, L4T R36) / `dvfs.DryRunBackend` |
| energymon | `energy.EnergyMeter` (INA3221 hwmon: VDD_IN, else sum of rails) |
| SysV shm + `ftok("/home/nvidia/...")` | `cohort.SharedCohort` (POSIX shared memory + `flock`) / `cohort.LocalCohort` |
| `generate_hashtable.py` | `profile.py` (`python -m neuos_modern.profile`) |
| `#define MAX_ACC` / `MIN_ENG` | `mode="MAX_ACC"` / `"MIN_ENG"` / `"DEFAULT"` |
| config line `ID CPU_freq CPU_mask[8] GPU_freq mem_freq` | same format, any mask length; on Orin mem_freq is the EMC rate in kHz |

Adapters (`neuos_modern/adapters/`):
- `torch_layers.py`: every leaf `nn.Module` call is a stage (closest to the original);
  `attach(model, ctl, twins=...)`, then `with ctl_job(model): model(x)`.
- `openai_stream.py`: llama.cpp `llama-server` / `vllm serve` `/v1/completions` streaming.
  Stage 0 = request + first token, the only approximable stage (approximate = send to the next
  lower tier's endpoint); stage k = the next `chunk_tokens` tokens.
- `pipeline.py`: cascades (ASR -> LLM -> TTS), one stage per component, each with its own ladder.

## Deviations (all documented in code)

- **Slot index modulo 10.** The original `__sync_fetch_and_add` + CAS(10 -> 0) can return an
  index >= 10 under a race; the port takes it modulo `SHM_SIZE`.
- **Lower clamp at 0.** The original only clamps the scheduled configuration to
  `configurations.size()-1`; a negative result would index out of bounds in C++ and wrap in
  Python, so the port also clamps at 0.
- **Empty-cohort guard.** When no cohort slot is non-zero the original computes
  `priority = 0/0` (NaN); the port keeps the previous priority.
- **Extra stages.** Stages beyond the warmup job's length reuse the last deadline and
  Uncertainty value (variable-length LLM generation).
- **One run per stage.** The original runs every layer `iteration = 20` times and divides; the
  port runs each stage once.
- **`variant="corrected"`** (opt-in). The default `"original"` reproduces a sign quirk of
  `schedule.h`: `slack += measured_exec - previous_expected_deadline`, i.e. an overrun *adds*
  slack, and the slack persists across jobs. `"corrected"` uses `slack -= measured - expected`
  and resets the slack to `period_ms*1000 - sum(deadlines)` at the start of every job.
  Everything else is identical.
- The profiler does not round energies to 2 decimals (as `generate_hashtable.py` does) and
  averages the `repeats` runs.

Quirks deliberately kept: the history check tests `current` but stores `prev_current`;
history keys are `std::to_string(float)` (`"%f"`); float->int truncates toward zero; knobs are
diffed against `configs[cfg[i-1]]` (the table entry, not the actual hardware state); cohort
slots are never cleared; MAX_ACC / MIN_ENG take the max / min over all 10 slots, zeros included;
the warmup job can approximate (A* is computed there too); `sched_us` accumulates over the job
and is added to the measured latency passed to `schedule()`.

## Usage

```python
from neuos_modern import *
ctl = NeuOSController(load_configs("orin.config"), load_uncertainty("Uncertainty.txt"),
                      period_ms=200, backend=OrinSysfsBackend(), cohort=SharedCohort("neuos"),
                      mode="DEFAULT", variant="original", energy=EnergyMeter().start())
job = ctl.begin_job(n_stages)
for i in range(n_stages):
    approx = job.before_stage(i, approximable=...)
    ...                                   # run stage i (lower tier if approx)
    job.after_stage(i)
log = job.end()                           # per-stage latency_us, energy_mj, config, approximated, slack, priority, a_star
```

### Profile (Uncertainty)

`job_fn(ctl)` is any function that runs one job through the controller API (all adapters do):

```sh
python -m neuos_modern.profile --runner mymod:job_fn --configs orin.config --every 10 \
    --repeats 3 --csv profile.csv --uncertainty Uncertainty.txt
python -m neuos_modern.profile --from-csv profile.csv --uncertainty Uncertainty.txt
```

Uncertainty per stage = max over configs of normEnergy / normLatency (each normalised by its
max over configs), written as one tab-separated line like the original.

### Run on Jetson Orin

DVFS and EMC writes (debugfs) need root. The config space can be built from sysfs with
`OrinSysfsBackend().enumerate_states()` + `build_config_space(...)`, or `--enumerate`:

```sh
sudo python examples/run_llm.py --configs orin.config --uncertainty Uncertainty.txt \
    --period-ms 4000 --ladder examples/ladders.json --prompts prompts.txt --jobs 50 \
    --mode DEFAULT --variant original --out neuos_llm.jsonl     # add --dry-run to not touch DVFS
```

**Restore semantics.** `OrinSysfsBackend()` snapshots the knob state at construction
(`snapshot()`: per cpufreq policy governor / `scaling_min_freq` / `scaling_max_freq`, the
`cpuN/online` flags, GPU devfreq `min_freq` / `max_freq`, EMC `clk/emc/rate` and
`mrq_rate_locked`). `restore()` writes exactly that state back (cpus online first, then
offline; min/max written in an order that keeps min <= max; EMC rate before the lock), so the
power mode's own caps (e.g. Orin Nano 15W: CPU max 1497.6 MHz, GPU max 612 MHz, EMC locked at
2133 MHz) come back, not the full hardware range. A process that starts while another one is
already actuating should pass a saved snapshot: `OrinSysfsBackend(baseline=json.load(...))`.
`run_llm.py` and `orin_eval.py` call `restore()` in `finally`; `orin_eval.py` also turns
SIGTERM / SIGINT into a clean exit through that `finally`. Shared cohort blocks persist like
SysV shm; remove them with `SharedCohort.unlink("neuos")`.

**Config space and power-mode caps.** `enumerate_states()` (default `capped=True`) keeps only
CPU frequencies <= the baseline `scaling_max_freq` and GPU frequencies <= the baseline devfreq
`max_freq` (cpufreq and devfreq both accept writes above the nvpmodel caps, checked on the
Nano). EMC rates are the debugfs `possible_rates`, filtered by `probe_emc_rates()`, which
writes each rate and keeps it only if `clk/emc/rate` reads back the same value (on the AGX
Orin in MAXN every EMC write is silently ignored and the rate stays at 3199 MHz, so only 3199
MHz remains). `subsample(values, n)` picks n evenly spaced levels including min and max.

**Energy.** `EnergyMeter` integrates `VDD_IN` when the board has it (Orin Nano: module input),
otherwise the sum of every INA3221 rail except the "sum of shunt voltages" channel (AGX Orin:
`VDD_GPU_SOC + VDD_CPU_CV + VIN_SYS_5V0 + VDDQ_VDD2_1V8AO`, ~8.5 W idle; Nano VDD_IN ~5 W idle).
`mark()` does not read the rails (an I2C transfer per rail); it extends the background
integral with the last sampled power, so per-layer stages stay cheap. DVFS writes to a
cpufreq policy / cpu `online` file that fail (e.g. `EBUSY` while a concurrent client
hot-plugs the same cpus) warn and increment `backend.write_errors` instead of raising.

### Orin Nano / AGX Orin driver: `examples/orin_eval.py`

```sh
PY=/path/to/venv/bin/python        # must run as root: sudo -n $PY ...
sudo -n $PY examples/orin_eval.py baseline --out base.json          # print + save the snapshot
# LLM: two llama-server tiers (full 1.5B on :18081, approx 0.5B on :18082)
sudo -n $PY examples/orin_eval.py profile --workload llm --out-dir R/llm --repeats 3
sudo -n $PY examples/orin_eval.py run --workload llm --profile-dir R/llm --jobs 30 \
    --mode DEFAULT --variant original --period-factor 1.3 --out R/llm/default_original.jsonl
# torch: resnet18 (random weights, 1x3x224x224, CUDA), SVD low-rank twins of every KxK conv
sudo -n $PY examples/orin_eval.py profile --workload torch --out-dir R/torch --cpu-levels 2 --cores all
sudo -n $PY examples/orin_eval.py run --workload torch --profile-dir R/torch --mode MAX_ACC --out ...
$PY examples/orin_eval.py summary R/llm/*.jsonl                    # markdown table rows
```

`profile` builds the grid (default: CPU 3 levels incl. max, cores {all, half}, GPU 4 levels
incl. max, EMC {665.6, 2133, 3199} MHz minus rates that do not read back), writes
`configs.txt`, `profile.csv`, `Uncertainty.txt` and `grid.json` (grid, caps, rails, and the
median wall latency of warm jobs at the fastest config). `run` loads them, sets
`period = --period-factor x that latency` (or `--period-ms`), and appends one JSON line per job
(wall / stage-sum latency, deadline met, energy, approximation rate, per-stage configs and
approximation flags, slack, priority, the cohort slots). Concurrent clients share a cohort via
the same `--cohort-prefix`; give them the same `--baseline` file. Measured results are in
`results/` (see `results/README.md`).

The torch adapter's twins come from `adapters.torch_layers.lowrank_twins(model, rank)`: like
`models/lowrank` in the original (Jaderberg et al. scheme 2), every KxK conv becomes a vertical
Kx1 conv to `r` channels and a horizontal 1xK conv, from the SVD of the weight reshaped to
`(Cin*K, Cout*K)`; `rank` is an int, a ratio of the full rank, or `{layer: r}` like
`models/lowrank/config.json`.

### Tests

```sh
cd modern && python -m pytest -q     # torch test is skipped when torch is missing
```
