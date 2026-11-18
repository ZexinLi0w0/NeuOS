"""NeuOS over an OpenAI-compatible streaming /v1/completions endpoint
(llama.cpp ``llama-server`` or ``vllm serve``).

Stage 0 = request + first token (prefill); it is the only approximable stage and
approximation sends the request to the next lower tier of the ladder (each tier is
``{"name", "url", "model"}``). Stage k >= 1 = the next ``chunk_tokens`` streamed
tokens, with a decision point (before_stage) before reading each chunk. One SSE
event with non-empty text counts as one token. If the stream ends exactly on a
chunk boundary the last stage reads zero tokens.
"""
import json
import math


def _sse_texts(resp):
    for raw in resp.iter_lines():
        if not raw or not raw.startswith(b"data:"):
            continue
        payload = raw[5:].strip()
        if payload == b"[DONE]":
            return
        choices = json.loads(payload).get("choices") or [{}]
        text = choices[0].get("text") or ""
        if text:
            yield text


def stream_completion(ctl, ladder, prompt, max_tokens=128, chunk_tokens=8, session=None,
                      timeout=600, extra_body=None):
    """Run one job; returns ``(text, log)``."""
    if session is None:
        import requests as session
    job = ctl.begin_job(1 + math.ceil(max(max_tokens - 1, 0) / chunk_tokens))

    tier = ladder.select(job.before_stage(0, ladder.approximable))
    body = {"model": tier["model"], "prompt": prompt, "max_tokens": max_tokens, "stream": True}
    body.update(extra_body or {})
    resp = session.post(tier["url"].rstrip("/") + "/v1/completions", json=body, stream=True, timeout=timeout)
    with resp:
        resp.raise_for_status()
        texts = _sse_texts(resp)
        first = next(texts, None)
        job.after_stage(0)
        out, done, i = [first or ""], first is None, 1
        while not done:
            job.before_stage(i, False)
            n = 0
            for t in texts:
                out.append(t)
                n += 1
                if n == chunk_tokens:
                    break
            else:
                done = True
            job.after_stage(i)
            i += 1
    log = job.end()
    log.update(tier=tier.get("name", tier["model"]), tokens=len(out) - (first is None))
    return "".join(out), log
