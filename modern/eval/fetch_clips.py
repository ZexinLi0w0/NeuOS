"""Build the LibriSpeech clip file the ASR task reads (``clips_npz`` in the board config).

73 validation-clean utterances of ``hf-internal-testing/librispeech_asr_dummy``, resampled to
16 kHz, stored as one npz: ``ids``, ``texts``, ``lens`` and the concatenated ``audio`` (float32).
Same format as the AudiOS repo's rtaudio cache (vendored here so the harness is self-contained).
Needs ``datasets``, ``soundfile`` and ``librosa`` and network access; run it once anywhere and
copy the file to the board. Without the file the ASR task falls back to synthetic audio.

    python fetch_clips.py --out /experiment/zexin/neuos_eval/data/librispeech_dummy.npz
"""
import argparse
import io
import os

import numpy as np


def fetch(out):
    import librosa
    import soundfile as sf
    from datasets import Audio, load_dataset
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    ds = ds.cast_column("audio", Audio(decode=False))   # decode with soundfile
    audio, ids, texts = [], [], []
    for ex in ds:
        a, src = sf.read(io.BytesIO(ex["audio"]["bytes"]), dtype="float32")
        if src != 16000:
            a = librosa.resample(a, orig_sr=src, target_sr=16000)
        audio.append(a.astype("float32"))
        ids.append(ex["id"])
        texts.append(ex["text"])
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    np.savez(out, ids=np.array(ids), texts=np.array(texts), lens=np.array([len(a) for a in audio]),
             audio=np.concatenate(audio))
    print("saved %d clips to %s" % (len(audio), out))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    fetch(ap.parse_args().out)
