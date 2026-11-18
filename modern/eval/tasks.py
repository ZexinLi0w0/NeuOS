"""One task process of the harness (ASR, perception, or LLM+TTS turn). Started by harness.py.

    python tasks.py <asr|perception|turn> --board-config C --run-dir R --policy neuos|default
    python tasks.py <task> --board-config C --profile-dir P --profile      # NeuOS Uncertainty profile

Policies
  neuos   : one NeuOSController per process (same config space / DVFS states, saved baseline),
            all processes in one SharedCohort. Approximation = the task's next lower tier
            (ASR, LLM) or SVD low-rank twins of the UNet convs (perception, per layer).
  default : top tiers only, no controller, no DVFS writes (stock governors).

Run protocol (files in the run dir): the task loads its models, warms them up, writes
``loaded_<task>``; waits for ``go_warmup`` then runs its NeuOS warmup job (first job of the
controller: random config, stage deadlines measured), writes ``ready_<task>``; waits for
``start.json`` {t_start, t_end}. ASR and perception releases are periodic on a grid anchored at
t_start; a release that occurs while the previous job still runs is dropped. ASR releases only
inside Listen phases (``phase.json``, written by the turn task, which conducts the trace).
"""
import argparse
import math
import os
import re
import sys
import time
from pathlib import Path

from common import (Jsonl, asr_pair, install_signal_handlers, llm_pair, load_cfg, load_clips,
                    mem_avail_mb, neuos_path, read_json, sleep_until, stage_summary, synthetic_clip,
                    twins_cache_path, write_json_atomic)

os.environ.setdefault("HF_HUB_OFFLINE", "1")
TASK_SEED = {"asr": 1, "perception": 2, "turn": 3}
SYS = "You are a helpful voice assistant. Reply to what the user said in one or two short sentences."
PROMPTS = [
    "What is a good way to start learning to play the guitar?",
    "Can you remind me how long to boil an egg for a soft yolk?",
    "Explain in simple terms how a CPU cache works.",
    "What should I pack for a two day hiking trip?",
    "How do I convert Fahrenheit to Celsius?",
    "Tell me a fun fact about octopuses.",
    "What are the main differences between TCP and UDP?",
    "Give me a quick tip for falling asleep faster.",
    "How far is the Moon from the Earth?",
    "Suggest a simple vegetarian dinner I can cook tonight.",
    "Why is the sky blue?",
    "What is the capital of Australia, and why is it not Sydney?",
]
SENT_END = re.compile(r"[.!?](\s|$)")


def now():
    return time.time()


# --- policy "default": no controller ------------------------------------------
class PassThrough:
    """Stand-in for NeuOSController under the default policy: never approximates, never
    touches DVFS; records per-stage latency (and energy when a meter is given)."""

    def __init__(self, energy=None):
        self.energy = energy

    def begin_job(self, n_stages):
        return _PTJob(self.energy)


class _PTJob:
    def __init__(self, energy):
        self.energy, self.stages = energy, []

    def before_stage(self, i, approximable=False):
        if self.energy:
            self.energy.mark()
        self._t0 = time.perf_counter()
        return False

    def after_stage(self, i):
        lat = int((time.perf_counter() - self._t0) * 1e6)
        mj = self.energy.mark() if self.energy else 0.0
        self.stages.append({"stage": i, "config": -1, "approximated": False, "a_star": 0,
                            "priority": 0.0, "latency_us": lat, "energy_mj": mj, "slack": 0.0})

    def end(self):
        return {"warmup": False, "slot": -1, "n_stages": len(self.stages), "sched_us": 0,
                "slack": 0.0, "latency_us": sum(s["latency_us"] for s in self.stages),
                "energy_mj": sum(s["energy_mj"] for s in self.stages), "stages": self.stages}


def timed_backend(base):
    """Harness-side instrumentation of the NeuOS backend (the package is not modified): counts
    apply() calls, knob writes and the time spent actuating DVFS."""
    class Timed(base):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.stats = {"apply": 0, "apply_ms": 0.0, "cpu_mask": 0, "cpu_freq": 0, "gpu_freq": 0, "emc_freq": 0}

        def apply(self, prev, cfg):
            t0 = time.perf_counter()
            super().apply(prev, cfg)
            self.stats["apply"] += 1
            self.stats["apply_ms"] += (time.perf_counter() - t0) * 1e3

        def set_cpu_mask(self, mask):
            self.stats["cpu_mask"] += 1
            super().set_cpu_mask(mask)

        def set_cpu_freq(self, khz):
            self.stats["cpu_freq"] += 1
            super().set_cpu_freq(khz)

        def set_gpu_freq(self, hz):
            self.stats["gpu_freq"] += 1
            super().set_gpu_freq(hz)

        def set_emc_freq(self, khz):
            self.stats["emc_freq"] += 1
            super().set_emc_freq(khz)
    return Timed


class Runtime:
    """Builds the controller for one task process and restores DVFS on close()."""

    def __init__(self, cfg, policy, task, period_ms, run_dir=None, profile_dir=None, baseline=None):
        from neuos_modern.energy import EnergyMeter
        self.backend = self.cohort = None
        self.energy = EnergyMeter(period_s=cfg["neuos"]["energy_period_s"]).start()
        if policy != "neuos":
            self.ctl = PassThrough(self.energy)
            return
        from neuos_modern import (NeuOSController, OrinSysfsBackend, SharedCohort, load_configs,
                                  load_uncertainty)
        nc = cfg["neuos"]
        self.backend = timed_backend(OrinSysfsBackend)(baseline=baseline)
        self.cohort = SharedCohort(nc["cohort_prefix"])
        configs = load_configs(os.path.join(profile_dir, "configs.txt"))
        unc = load_uncertainty(os.path.join(profile_dir, "Uncertainty_%s.txt" % task))
        self.ctl = NeuOSController(configs, unc, period_ms, self.backend, self.cohort,
                                   mode=nc["mode"], variant=nc["variant"],
                                   seed=nc["seed"] + TASK_SEED[task], energy=self.energy)

    def dvfs_delta(self):
        """DVFS actuation by this process since the previous call (neuos only)."""
        if self.backend is None:
            return None
        cur = dict(self.backend.stats)
        prev, self._prev = getattr(self, "_prev", None) or {k: 0 for k in cur}, cur
        return {k: round(cur[k] - prev[k], 2) for k in cur}

    def close(self):
        self.energy.stop()
        if self.backend is not None:
            self.backend.restore()
        if self.cohort is not None:
            self.cohort.close()


# --- tasks ---------------------------------------------------------------------
class ASRTask:
    """Whisper; one job = stage 0 (tier decision + feature extraction + encoder) then greedy
    decoder chunks of ``chunk_tokens`` tokens (decision point before each)."""
    name = "asr"

    def __init__(self, cfg, policy):
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        self.torch, self.cfg, self.A = torch, cfg, cfg["asr"]
        full, approx = asr_pair(cfg)
        names = [full] + ([approx] if policy == "neuos" and approx != full else [])
        self.tiers = []
        for n in names:
            proc = WhisperProcessor.from_pretrained(n)
            m = WhisperForConditionalGeneration.from_pretrained(n, torch_dtype=torch.float16).cuda().eval()
            tok = proc.tokenizer
            prefix = tok.convert_tokens_to_ids(["<|startoftranscript|>", "<|en|>", "<|transcribe|>",
                                                "<|notimestamps|>"])
            self.tiers.append({"name": n.split("/")[-1], "proc": proc, "model": m, "prefix": prefix,
                               "eos": tok.eos_token_id})
        self.approximable = len(self.tiers) > 1
        self.n_stages = 1 + math.ceil(self.A["max_new_tokens"] / self.A["chunk_tokens"])
        clips = load_clips(cfg.get("clips_npz"))
        self.audio_source = "librispeech" if clips else "synthetic"
        self.clips = clips or [synthetic_clip(20, s) for s in range(8)]

    def utterance(self, clip_idx, seconds):
        """Audio heard so far in a Listen phase: clips concatenated from clip_idx, cut to
        ``seconds`` (>= 0.5 s, <= window_max_s)."""
        import numpy as np
        need = int(16000 * min(max(seconds, 0.5), self.A["window_max_s"]))
        parts, k, n = [], clip_idx, 0
        while n < need:
            c = self.clips[k % len(self.clips)]
            parts.append(c)
            n += len(c)
            k += 1
        return np.concatenate(parts)[:need]

    def job(self, ctl, audio):
        torch = self.torch
        job = ctl.begin_job(self.n_stages)
        with torch.inference_mode():
            t = self.tiers[1 if job.before_stage(0, self.approximable) else 0]
            m = t["model"]
            feats = t["proc"](audio, sampling_rate=16000, return_tensors="pt").input_features
            enc = m.get_encoder()(feats.cuda().half()).last_hidden_state
            torch.cuda.synchronize()
            job.after_stage(0)
            ids = torch.tensor([t["prefix"]], device="cuda")
            past, out, done, i = None, [], False, 1
            while not done and i < self.n_stages:
                job.before_stage(i, False)
                for _ in range(self.A["chunk_tokens"]):
                    o = m(encoder_outputs=(enc,), decoder_input_ids=ids, past_key_values=past, use_cache=True)
                    past = o.past_key_values
                    nxt = int(o.logits[0, -1].argmax())
                    if nxt == t["eos"] or len(out) >= self.A["max_new_tokens"]:
                        done = True
                        break
                    out.append(nxt)
                    ids = torch.tensor([[nxt]], device="cuda")
                torch.cuda.synchronize()
                job.after_stage(i)
                i += 1
        log = job.end()
        text = t["proc"].tokenizer.decode(out, skip_special_tokens=True).strip()
        return log, {"tier": t["name"], "n_tok": len(out), "audio_s": round(len(audio) / 16000, 2),
                     "text": text[:80]}

    def warm(self):
        a = self.utterance(0, 5)
        for t in range(len(self.tiers)):
            for _ in range(2):
                self.job(_Force(t == 1), a)

    def profile_job(self, rec):
        self.job(rec, self.utterance(0, 10))


class _Force:
    """Controller stub that forces the approximation decision (to warm up twins / tiers)."""

    def __init__(self, approx):
        self.approx = approx

    def begin_job(self, n):
        return self

    def before_stage(self, i, approximable=False):
        return self.approx and approximable

    def after_stage(self, i):
        pass

    def end(self):
        return {}


class PerceptionTask:
    """SD x4-upscaler UNet, one denoising step per frame (as frame_loop.py in the AudiOS
    repo's motivation/scripts).
    neuos: every leaf module call is a stage (torch_layers adapter), the KxK convs have SVD
    low-rank twins. default: plain forward."""
    name = "perception"

    def __init__(self, cfg, policy):
        import torch
        from diffusers import UNet2DConditionModel
        self.torch, self.cfg, self.P = torch, cfg, cfg["perception"]
        self.policy = policy
        H = self.P["latent_hw"]
        u = UNet2DConditionModel.from_pretrained(self.P["model"], subfolder="unet", variant="fp16",
                                                 torch_dtype=torch.float16).cuda().eval()
        g = torch.Generator(device="cuda").manual_seed(0)
        self.inp = (torch.randn(1, u.config.in_channels, H, H, device="cuda", dtype=torch.float16, generator=g),
                    torch.full((1,), 500, device="cuda", dtype=torch.long))
        self.kw = {"encoder_hidden_states": torch.randn(1, 77, u.config.cross_attention_dim, device="cuda",
                                                        dtype=torch.float16, generator=g),
                   "class_labels": torch.full((1,), 20, device="cuda", dtype=torch.long)}
        self.unet, self.neu, self.twins = u, None, {}
        if policy == "neuos" or policy == "profile":
            from neuos_modern.adapters.torch_layers import attach
            if policy == "neuos":
                twins = torch.load(twins_cache_path(cfg), weights_only=False)
                self.twins = {k: v.cuda().eval() for k, v in twins.items()}
            self.neu = attach(u, PassThrough(), self.twins)
        self.approximable = len(self.twins)

    def _fwd(self):
        return self.unet(*self.inp, **self.kw).sample

    def job(self, ctl):
        torch = self.torch
        with torch.inference_mode():
            if self.neu is None:            # default policy: no per-layer hooks
                j = ctl.begin_job(1)
                j.before_stage(0)
                self._fwd()
                torch.cuda.synchronize()
                j.after_stage(0)
                log = j.end()
            else:
                self.neu.ctl = ctl
                with self.neu.job():
                    self._fwd()
                log = dict(self.neu.last_log or {})
        st = log.get("stages") or []
        n_apx = sum(bool(s["approximated"]) for s in st)
        return log, {"tier": "twins" if n_apx else "full", "n_twin_stages": n_apx,
                     "n_approximable": self.approximable}

    def warm(self):
        with self.torch.inference_mode():
            for _ in range(5):
                self._fwd()
        for _ in range(3):
            self.job(PassThrough())           # sets neu.n_stages to the observed count
        if self.neu is not None and self.twins:
            for _ in range(3):
                self.job(_ForceJob())

    def profile_job(self, rec):
        self.job(rec)


class _ForceJob(_Force):
    def __init__(self):
        super().__init__(True)

    def end(self):
        return {"stages": []}


class TurnTask:
    """LLM + TTS turn. Stage 0 = tier decision + request + first token (llama-server
    /v1/completions stream; approx = the next lower tier's server); stage k = next
    ``chunk_tokens`` tokens; as soon as the streamed text completes its first sentence, a TTS
    stage synthesises it (first audio); after the stream ends, a last stage synthesises the rest.
    TTS (MMS-TTS) has one tier and is never approximated."""
    name = "turn"

    def __init__(self, cfg, policy):
        import requests
        import torch
        from transformers import AutoTokenizer, VitsModel
        self.torch, self.cfg, self.L = torch, cfg, cfg["llm"]
        full, approx = llm_pair(cfg)
        p = self.L["ports"]
        self.tiers = [dict(full, url="http://127.0.0.1:%d" % p[0])]
        if policy == "neuos" and approx is not full:
            self.tiers.append(dict(approx, url="http://127.0.0.1:%d" % p[1]))
        self.approximable = len(self.tiers) > 1
        self.session = requests.Session()
        self.ttok = AutoTokenizer.from_pretrained(cfg["tts"]["model"])
        # tts.device "cpu": MMS-TTS on the CPU; this process then never touches CUDA, so it has no
        # CUDA context (~1 GB of MemAvailable on the Orin Nano)
        self.dev = cfg["tts"].get("device", "cuda")
        self.tts = VitsModel.from_pretrained(cfg["tts"]["model"]).to(self.dev).eval()
        self.sr = self.tts.config.sampling_rate
        n_chunks = math.ceil(max(self.L["max_tokens"] - 1, 0) / self.L["chunk_tokens"])
        self.n_stages = 1 + n_chunks + 2

    @staticmethod
    def prompt(text):
        return ("<|im_start|>system\n%s<|im_end|>\n<|im_start|>user\n%s<|im_end|>\n"
                "<|im_start|>assistant\n" % (SYS, text))

    def speak(self, text):
        torch = self.torch
        if not text.strip():
            return 0.0, 0.0
        t0 = time.perf_counter()
        with torch.inference_mode():
            torch.manual_seed(0)
            x = self.ttok(text, return_tensors="pt").to(self.dev)
            wav = self.tts(**x).waveform
            if self.dev == "cuda":
                torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        return dt, wav.shape[-1] / self.sr

    def _events(self, resp):
        import json
        for raw in resp.iter_lines():
            if not raw or not raw.startswith(b"data:"):
                continue
            payload = raw[5:].strip()
            if payload == b"[DONE]":
                return
            ev = json.loads(payload)
            if ev.get("timings"):
                self._timings = ev["timings"]
            ch = ev.get("choices") or [{}]
            text = ch[0].get("text") or ""
            if text:
                yield text

    def job(self, ctl, user_text, t_request_end=None, ignore_eos=None):
        """Returns (log, info). info["first_audio_t"] is the wall time the first audio is ready."""
        L = self.L
        self._timings = None
        job = ctl.begin_job(self.n_stages)
        tier = self.tiers[1 if job.before_stage(0, self.approximable) else 0]
        body = {"model": tier["name"], "prompt": self.prompt(user_text), "max_tokens": L["max_tokens"],
                "stream": True, "temperature": L["temperature"],
                "ignore_eos": L["ignore_eos"] if ignore_eos is None else ignore_eos,
                "cache_prompt": False}
        t_send = now()
        resp = self.session.post(tier["url"] + "/v1/completions", json=body, stream=True, timeout=300)
        tts_calls, first_audio_t, spoken_upto = [], None, None
        with resp:
            resp.raise_for_status()
            ev = self._events(resp)
            first = next(ev, None)
            t_first_tok = now()
            job.after_stage(0)
            out, done, i = [first or ""], first is None, 1
            tok_t = [t_first_tok]
            while not done:
                job.before_stage(i, False)
                n = 0
                for t in ev:
                    out.append(t)
                    tok_t.append(now())
                    n += 1
                    if n == L["chunk_tokens"]:
                        break
                else:
                    done = True
                job.after_stage(i)
                i += 1
                if spoken_upto is None:
                    m = SENT_END.search("".join(out))
                    if m:                                   # first sentence complete: speak it now
                        spoken_upto = m.end()
                        job.before_stage(i, False)
                        dt, dur = self.speak("".join(out)[:spoken_upto])
                        job.after_stage(i)
                        i += 1
                        first_audio_t = now()
                        tts_calls.append({"kind": "first", "synth_s": dt, "audio_s": dur})
        reply = "".join(out)
        job.before_stage(i, False)
        if spoken_upto is None:                             # no sentence end: the whole reply is the first audio
            dt, dur = self.speak(reply)
            first_audio_t = now()
            tts_calls.append({"kind": "first", "synth_s": dt, "audio_s": dur})
        else:
            dt, dur = self.speak(reply[spoken_upto:])
            if dur:
                tts_calls.append({"kind": "rest", "synth_s": dt, "audio_s": dur})
        job.after_stage(i)
        log = job.end()
        n_tok = len(out) - (first is None)
        tim = self._timings or {}
        client_tps = None
        if n_tok > 1:
            gen_s = tok_t[-1] - tok_t[0] - sum(c["synth_s"] for c in tts_calls if c["kind"] == "first"
                                               and spoken_upto is not None)
            client_tps = (n_tok - 1) / gen_s if gen_s > 0 else None
        info = {"tier": tier["name"], "n_tok": n_tok, "t_send": t_send, "ttft_s": t_first_tok - t_send,
                "first_audio_t": first_audio_t, "tps_server": tim.get("predicted_per_second"),
                "prompt_ms_server": tim.get("prompt_ms"), "tps_client": client_tps,
                "tts": tts_calls, "reply": reply[:160]}
        return log, info

    def warm(self):
        for t in range(len(self.tiers)):
            self.job(_Force(t == 1), PROMPTS[0])

    def profile_job(self, rec):
        self.job(rec, PROMPTS[2], ignore_eos=True)     # all max_tokens: profiles every stage index


TASKS = {"asr": ASRTask, "perception": PerceptionTask, "turn": TurnTask}


# --- run -----------------------------------------------------------------------
def release_host_memory():
    """After loading + warm-up: drop the CPU-side copies of the checkpoints (freed by .cuda() but
    kept in the glibc heap) and the CUDA caching allocator's free blocks. On Jetson the GPU and
    the CPU share DRAM, so both count against MemAvailable."""
    import ctypes
    import gc
    import torch
    gc.collect()
    cuda = torch.cuda.is_initialized()          # never create a CUDA context here
    if cuda:
        torch.cuda.empty_cache()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass
    with open("/proc/self/status") as f:
        rss = next((int(l.split()[1]) // 1024 for l in f if l.startswith("VmRSS")), None)
    return {"rss_mb": rss, "cuda_context": cuda,
            "cuda_alloc_mb": torch.cuda.memory_allocated() >> 20 if cuda else 0,
            "cuda_reserved_mb": torch.cuda.memory_reserved() >> 20 if cuda else 0}


def wait_file(path, stop_at=None):
    while not os.path.exists(path):
        if stop_at and now() > stop_at:
            raise TimeoutError(path)
        time.sleep(0.1)


def periodic_loop(task, rt, run, cfg, log, t_start, t_end, floor):
    """Periodic releases (ASR: only inside Listen phases); releases that fall while the
    previous job is running are dropped."""
    tc = cfg[task.name]
    P, D = tc["period_ms"] / 1e3, tc["deadline_ms"] / 1e3
    k = 0

    def listen_at(r):
        ph = read_json(run / "phase.json", {}) or {}
        if ph.get("phase") == "listen" or ph.get("until"):
            if ph.get("since", 1e30) <= r < ph.get("until", -1):
                return ph
        return None

    while True:
        rel = t_start + k * P
        if rel >= t_end:
            break
        sleep_until(rel)
        ph = None
        if task.name == "asr":
            ph = listen_at(rel)
            if ph is None:
                k += 1
                continue
        av = mem_avail_mb()
        if av < floor:
            log.write(ev="abort", reason="mem_floor", avail_mb=av, t=now())
            return "mem_floor"
        s = now()
        if task.name == "asr":
            jl, info = task.job(rt.ctl, task.utterance(ph["clip"], rel - ph["since"]))
            info["turn"] = ph["turn"]
        else:
            jl, info = task.job(rt.ctl)
        e = now()
        lat = e - rel
        log.write(ev="job", job=k, release=rel, start=s, end=e, latency_ms=lat * 1e3, deadline_ms=D * 1e3,
                  status="met" if lat <= D else "late", avail_mb=av, **info, neuos=stage_summary(jl),
                  dvfs=rt.dvfs_delta())
        k += 1
        while t_start + k * P < min(e, t_end):             # released while the job was running
            r = t_start + k * P
            if task.name != "asr" or listen_at(r):
                log.write(ev="job", job=k, release=r, start=None, end=None, latency_ms=None,
                          deadline_ms=D * 1e3, status="dropped")
            k += 1
    return "done"


def conductor_loop(task, rt, run, cfg, log, t_start, t_end, floor):
    """The turn task conducts the Listen/Respond trace: Listen phase (ASR active) for listen_s,
    then a turn request released at the end of Listen, then idle_s after the turn finishes."""
    trace = read_json(run / "trace.json")["phases"]
    D = cfg["turn"]["deadline_ms"] / 1e3
    sleep_until(t_start)
    for n, ph in enumerate(trace):
        since = now()
        if since >= t_end:
            break
        until = min(since + ph["listen_s"], t_end)
        write_json_atomic(run / "phase.json", {"phase": "listen", "turn": n, "since": since,
                                               "until": until, "clip": ph["clip"]})
        log.write(ev="listen", turn=n, since=since, until=until, clip=ph["clip"])
        sleep_until(until)
        if until >= t_end:
            break
        write_json_atomic(run / "phase.json", {"phase": "respond", "turn": n, "since": until})
        av = mem_avail_mb()
        if av < floor:
            log.write(ev="abort", reason="mem_floor", avail_mb=av, t=now())
            return "mem_floor"
        s = now()
        jl, info = task.job(rt.ctl, PROMPTS[ph["prompt"] % len(PROMPTS)], until)
        e = now()
        fa = info["first_audio_t"]
        turn_lat = fa - until if fa else None
        log.write(ev="job", job=n, release=until, start=s, end=e, latency_ms=(e - until) * 1e3,
                  turn_latency_ms=turn_lat * 1e3 if turn_lat is not None else None, deadline_ms=D * 1e3,
                  status="met" if turn_lat is not None and turn_lat <= D else "late", avail_mb=av,
                  prompt=ph["prompt"], **info, neuos=stage_summary(jl), dvfs=rt.dvfs_delta())
        write_json_atomic(run / "phase.json", {"phase": "idle", "turn": n, "since": e})
        sleep_until(min(now() + ph["idle_s"], t_end))
    return "done"


def run_task(a, cfg):
    run = Path(a.run_dir)
    name = a.task
    log = Jsonl(run / ("%s.jsonl" % name))
    floor = cfg["mem_floor_mb"]
    rt = None
    try:
        av0 = mem_avail_mb()
        t0 = now()
        task = TASKS[name](cfg, a.policy)
        task.warm()
        mem = release_host_memory()
        log.write(ev="loaded", t=now(), load_s=now() - t0, avail_before_mb=av0, avail_mb=mem_avail_mb(), **mem,
                  policy=a.policy, approximable=getattr(task, "approximable", 0),
                  audio_source=getattr(task, "audio_source", None), pid=os.getpid())
        write_json_atomic(run / ("loaded_%s" % name), {"avail_mb": mem_avail_mb()})
        wait_file(run / "go_warmup")
        period = cfg["turn"]["deadline_ms"] if name == "turn" else cfg[name]["deadline_ms"]
        rt = Runtime(cfg, a.policy, name, period, run_dir=run, profile_dir=a.profile_dir,
                     baseline=read_json(run / "baseline.json"))
        # NeuOS warmup job (the controller's first job), outside the measured window
        s = now()
        if name == "asr":
            jl, info = task.job(rt.ctl, task.utterance(0, 5))
        elif name == "turn":
            jl, info = task.job(rt.ctl, PROMPTS[0])
        else:
            jl, info = task.job(rt.ctl)
        log.write(ev="warmup_job", start=s, end=now(), latency_ms=(now() - s) * 1e3,
                  tier=info.get("tier"), neuos=stage_summary(jl), dvfs=rt.dvfs_delta())
        write_json_atomic(run / ("ready_%s" % name), {"t": now()})
        wait_file(run / "start.json")
        st = read_json(run / "start.json")
        loop = conductor_loop if name == "turn" else periodic_loop
        res = loop(task, rt, run, cfg, log, st["t_start"], st["t_end"], floor)
        log.write(ev="end", t=now(), result=res, dvfs_write_errors=getattr(rt.backend, "write_errors", 0))
        return 3 if res == "mem_floor" else 0
    except SystemExit as e:
        log.write(ev="signal_exit", t=now(), code=str(e.code))
        raise
    except Exception as e:
        import traceback
        log.write(ev="error", t=now(), error=repr(e), tb=traceback.format_exc()[-2000:])
        oom = "out of memory" in repr(e).lower()
        if oom:
            log.write(ev="oom", t=now(), error=repr(e)[:300])
        return 2
    finally:
        if rt is not None:
            rt.close()                       # exact baseline (saved snapshot) back
        log.close()


def profile_task(a, cfg):
    """Offline Uncertainty profile of one task alone (neuos_modern.profile), full tier, no
    approximation; also the median warm wall latency at the fastest config."""
    import json
    import statistics
    from neuos_modern import OrinSysfsBackend, load_configs
    from neuos_modern.config import format_uncertainty
    from neuos_modern.energy import EnergyMeter
    from neuos_modern.profile import _Recorder, compute_uncertainty, profile, write_csv
    pdir = Path(a.profile_dir)
    name = a.task
    task = TASKS[name](cfg, "profile")
    task.warm()
    release_host_memory()
    configs = load_configs(pdir / "configs.txt")
    backend = OrinSysfsBackend(baseline=read_json(pdir / "baseline.json"))
    energy = EnergyMeter(period_s=cfg["neuos"]["energy_period_s"]).start()
    info = {"task": name, "avail_mb_loaded": mem_avail_mb()}
    try:
        backend.apply(None, configs[-1])
        walls = []
        for _ in range(3):
            t0 = time.perf_counter()
            task.profile_job(_Recorder(energy, time.perf_counter))
            walls.append((time.perf_counter() - t0) * 1e3)
        info["max_config_wall_ms"] = statistics.median(walls)
        t0 = now()
        rows = profile(task.profile_job, configs, backend, energy, cfg["profile"]["repeats"])
        info["profile_s"] = now() - t0
    finally:
        energy.stop()
        backend.restore()
    write_csv(pdir / ("profile_%s.csv" % name), rows)
    u = compute_uncertainty(rows)
    (pdir / ("Uncertainty_%s.txt" % name)).write_text(format_uncertainty(u))
    info["n_stages"] = len(u)
    (pdir / ("profile_%s.json" % name)).write_text(json.dumps(info, indent=1))
    print(json.dumps(info), flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=sorted(TASKS))
    ap.add_argument("--board-config", required=True)
    ap.add_argument("--policy", choices=("neuos", "default"), default="neuos")
    ap.add_argument("--run-dir")
    ap.add_argument("--profile-dir")
    ap.add_argument("--profile", action="store_true")
    a = ap.parse_args()
    cfg = load_cfg(a.board_config)
    if a.run_dir and os.path.exists(os.path.join(a.run_dir, "board_config.json")):
        cfg = load_cfg(os.path.join(a.run_dir, "board_config.json"))   # run-time overrides
    neuos_path(cfg)
    install_signal_handlers()
    return profile_task(a, cfg) if a.profile else run_task(a, cfg)


if __name__ == "__main__":
    sys.exit(main())
