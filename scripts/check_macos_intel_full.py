#!/usr/bin/env python3
"""Load the real OmniVoice checkpoint and synthesize a tiny CPU sample on Intel macOS."""

import math
import platform
import sys

import numpy as np
import torch

from omnivoice import OmniVoice

MODEL_ID = "k2-fsa/OmniVoice"


def main() -> None:
    print(f"python={sys.version.split()[0]}")
    print(f"machine={platform.machine()}")
    print(f"torch={torch.__version__}")
    print(f"loading={MODEL_ID}")

    if sys.platform != "darwin" or platform.machine() != "x86_64":
        raise RuntimeError("full Intel integration test must run on macOS x86_64")

    model = OmniVoice.from_pretrained(
        MODEL_ID,
        device_map="cpu",
        dtype=torch.float32,
    )

    if model.audio_tokenizer is None or model.text_tokenizer is None:
        raise RuntimeError("real checkpoint loaded without required tokenizers")
    if model.sampling_rate is None or model.sampling_rate <= 0:
        raise RuntimeError(f"invalid sampling rate: {model.sampling_rate}")
    if str(model.device) != "cpu":
        raise RuntimeError(f"expected CPU model placement, got {model.device}")

    print(
        "checkpoint-load=PASS "
        f"sampling_rate={model.sampling_rate} "
        f"audio_frame_rate={model.audio_tokenizer.config.frame_rate}"
    )

    # One diffusion step and only a handful of audio frames. This is not a
    # quality test; it verifies the actual Qwen3 + OmniVoice + Higgs execution
    # path works end-to-end with the Intel-compatible Torch 2.2 runtime.
    audio = model.generate(
        text="Hi.",
        language="en",
        instruct="female",
        duration=0.08,
        num_step=1,
        guidance_scale=0.0,
        position_temperature=0.0,
        class_temperature=0.0,
        postprocess_output=False,
        pad_duration=0.0,
        fade_duration=0.0,
    )

    if len(audio) != 1:
        raise RuntimeError(f"expected one generated item, got {len(audio)}")
    samples = np.asarray(audio[0])
    if samples.ndim != 1 or samples.size == 0:
        raise RuntimeError(f"invalid generated waveform shape: {samples.shape}")
    if not np.isfinite(samples).all():
        raise RuntimeError("generated waveform contains NaN/Inf")
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    if not math.isfinite(peak):
        raise RuntimeError(f"invalid generated waveform peak: {peak}")

    print(
        "end-to-end-generation=PASS "
        f"shape={samples.shape} samples={samples.size} peak={peak:.6f}"
    )


if __name__ == "__main__":
    main()
