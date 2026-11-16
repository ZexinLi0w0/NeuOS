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
| low-rank twin of a Convolution layer | next lower model tier (`tiers.TierLadder`), or a `twins` module in `adapters/torch_layers.py` |
| TX2 / AGX `setconfig.h` sysfs | `dvfs.OrinSysfsBackend` (Orin, L4T R36) / `dvfs.DryRunBackend` |
| energymon | `energy.EnergyMeter` (INA3221 hwmon, VDD_IN) |
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

`run_llm.py` calls `backend.restore()` in a `finally` block (all cores online, full CPU/GPU
ranges, EMC unlocked). Shared cohort blocks persist like SysV shm; remove them with
`SharedCohort.unlink("neuos")`.

### Tests

```sh
cd modern && python -m pytest -q     # torch test is skipped when torch is missing
```
