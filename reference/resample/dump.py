"""Resamples a chirp and white noise with torchaudio in float64, with the parameters torchaudio's documentation gives
for librosa's kaiser_best and, for the rates fish-speech resamples references between, with torchaudio's defaults, and
saves what checks/common/resample-check.cpp compares speech.cpp's resampler with.

usage: uv run python dump.py <out dir>

Writes <out dir>/<from>-<to>-<signal>/input.npy, float32 samples at <from> Hz, and output.npy, torchaudio's float64
output at <to> Hz, and the same in <from>-<to>-<signal>-hann/ with torchaudio's default Hann filter. Each input is 180697 samples long: torchaudio rounds the length of its output to float32 before
taking its ceiling, which for this length and 44100 to 16000 Hz is one sample below the exact ceiling.
"""

import argparse
import os

import numpy as np
import torch
import torchaudio.functional as F

KAISER_BEST = {"lowpass_filter_width": 64, "rolloff": 0.9475937167399596, "resampling_method": "sinc_interp_kaiser",
               "beta": 14.769656459379492}
RATES = [(16000, 48000), (48000, 16000), (44100, 16000), (22050, 24000), (8000, 16000), (44100, 48000), (24000, 48000),
         (11025, 16000)]
# The rates of recordings that fish-speech brings to Drama 3's 44100 Hz with torchaudio's defaults.
HANN_RATES = [(48000, 44100), (16000, 44100), (24000, 44100), (22050, 44100)]
LENGTH = 180697

parser = argparse.ArgumentParser()
parser.add_argument("out_dir")
args = parser.parse_args()


def chirp(rate):
    """A sine sweeping from 20 Hz to the Nyquist frequency of `rate`, so that a lowpass's whole band is crossed."""
    t = np.arange(LENGTH) / rate
    duration = LENGTH / rate
    f0, f1 = 20.0, rate / 2
    phase = 2 * np.pi * (f0 * t + (f1 - f0) * t * t / (2 * duration))
    return (0.5 * np.sin(phase)).astype(np.float32)


def noise(rate):
    return np.random.default_rng(rate).uniform(-0.5, 0.5, LENGTH).astype(np.float32)


cases = [(orig, new, KAISER_BEST, "") for orig, new in RATES] + [(orig, new, {}, "-hann") for orig, new in HANN_RATES]
for orig, new, parameters, suffix in cases:
    for name, make in (("chirp", chirp), ("noise", noise)):
        x = make(orig)
        y = F.resample(torch.from_numpy(x).double(), orig, new, **parameters).numpy()
        assert y.dtype == np.float64
        folder = os.path.join(args.out_dir, f"{orig}-{new}-{name}{suffix}")
        os.makedirs(folder, exist_ok=True)
        np.save(os.path.join(folder, "input.npy"), x)
        np.save(os.path.join(folder, "output.npy"), y)
        exact = -(-new * LENGTH // orig)
        print(f"{orig} -> {new} {name}: {len(x)} -> {len(y)} samples (exact ceiling {exact})")
