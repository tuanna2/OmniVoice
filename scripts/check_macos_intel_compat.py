#!/usr/bin/env python3
"""Smoke-test the forced macOS Intel OmniVoice runtime."""

import os
import platform
import sys
import tempfile

import torch
import transformers


def check(label, fn):
    try:
        fn()
    except Exception as exc:
        print(f"FAIL {label}: {type(exc).__name__}: {exc}")
        return False
    print(f"PASS {label}")
    return True


print(f"python={sys.version.split()[0]}")
print(f"platform={platform.platform()}")
print(f"machine={platform.machine()}")
print(f"torch={torch.__version__}")
print(f"transformers={transformers.__version__}")

if sys.platform != "darwin" or platform.machine() != "x86_64":
    print("WARNING: this compatibility probe is intended for macOS x86_64.")

checks = []


def check_versions_and_backend():
    if not torch.__version__.startswith("2.2.2"):
        raise RuntimeError(f"expected torch 2.2.2, got {torch.__version__}")
    if transformers.__version__ != "5.3.0":
        raise RuntimeError(f"expected transformers 5.3.0, got {transformers.__version__}")
    from transformers.utils.import_utils import is_torch_available

    if not is_torch_available():
        raise RuntimeError(
            "Transformers still reports PyTorch unavailable; torch 2.2 shim was not applied"
        )


checks.append(check("forced versions + Transformers torch backend", check_versions_and_backend))


def check_qwen3_runtime():
    from transformers import AutoModel, Qwen3Config

    cfg = Qwen3Config(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
    )
    cfg._attn_implementation = "eager"
    model = AutoModel.from_config(cfg).eval()
    with torch.inference_mode():
        out = model(input_ids=torch.tensor([[1, 2, 3]], dtype=torch.long))
    if out.last_hidden_state.shape != (1, 3, 32):
        raise RuntimeError(f"unexpected Qwen3 output shape: {out.last_hidden_state.shape}")


checks.append(check("Qwen3 tiny forward on Torch 2.2.2", check_qwen3_runtime))


def check_higgs():
    from transformers import HiggsAudioV2TokenizerConfig, HiggsAudioV2TokenizerModel

    cfg = HiggsAudioV2TokenizerConfig()
    if cfg.frame_rate <= 0 or cfg.hop_length <= 0:
        raise RuntimeError("invalid Higgs Audio V2 tokenizer config")
    if HiggsAudioV2TokenizerModel is None:
        raise RuntimeError("HiggsAudioV2TokenizerModel unavailable")


checks.append(check("Higgs Audio V2 tokenizer API", check_higgs))


def check_higgs_tiny_forward():
    # Mirrors Transformers' own Higgs tokenizer unit-test topology, but keeps
    # every dimension small enough to run quickly on the Intel CPU runner.
    from transformers import (
        DacConfig,
        HiggsAudioV2TokenizerConfig,
        HiggsAudioV2TokenizerModel,
        HubertConfig,
    )

    acoustic_cfg = DacConfig(
        decoder_hidden_size=8,
        encoder_hidden_size=8,
        codebook_size=16,
        downsampling_ratios=[16, 16],
    )
    semantic_cfg = HubertConfig(
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=12,
        conv_dim=(4, 4, 4, 4, 4, 4, 4),
    )
    cfg = HiggsAudioV2TokenizerConfig(
        sample_rate=16000,
        audio_channels=1,
        codebook_size=16,
        acoustic_model_config=acoustic_cfg,
        semantic_model_config=semantic_cfg,
    )
    model = HiggsAudioV2TokenizerModel(cfg).eval()
    input_values = torch.zeros((1, 1, 256), dtype=torch.float32)

    with torch.inference_mode():
        encoded = model.encode(input_values)
        if encoded.audio_codes is None or encoded.audio_codes.numel() == 0:
            raise RuntimeError("Higgs tiny encode returned no audio codes")
        decoded = model.decode(encoded.audio_codes)
        if decoded.audio_values is None or decoded.audio_values.numel() == 0:
            raise RuntimeError("Higgs tiny decode returned no audio")
        output = model(input_values=input_values)

    if output.audio_values.shape != input_values.shape:
        raise RuntimeError(
            f"unexpected Higgs tiny forward shape: {output.audio_values.shape} != {input_values.shape}"
        )


checks.append(check("Higgs tiny encode/decode/forward on Torch 2.2.2", check_higgs_tiny_forward))


def check_omnivoice_runtime():
    from transformers import Qwen3Config

    from omnivoice.models.omnivoice import OmniVoice, OmniVoiceConfig

    llm_cfg = Qwen3Config(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
    )
    llm_cfg._attn_implementation = "eager"
    cfg = OmniVoiceConfig(
        audio_vocab_size=16,
        audio_mask_id=15,
        num_audio_codebook=2,
        audio_codebook_weights=[1, 1],
        llm_config=llm_cfg,
    )
    model = OmniVoice(cfg).eval()
    if model.audio_embeddings.num_embeddings != 32:
        raise RuntimeError("unexpected OmniVoice audio embedding size")


checks.append(check("OmniVoice tiny model construction", check_omnivoice_runtime))


def check_prompt_roundtrip():
    from omnivoice.models.omnivoice import VoiceClonePrompt

    prompt = VoiceClonePrompt(
        ref_audio_tokens=torch.tensor([[1, 2], [3, 4]], dtype=torch.long),
        ref_text="intel smoke test",
        ref_rms=0.1,
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "prompt.pt")
        prompt.save(path)
        restored = VoiceClonePrompt.load(path)
    if restored.ref_text != prompt.ref_text:
        raise RuntimeError("VoiceClonePrompt roundtrip failed")


checks.append(check("VoiceClonePrompt torch.load compatibility", check_prompt_roundtrip))

if not all(checks):
    print(
        "\nThe forced Intel runtime installed but at least one runtime API is "
        "incompatible with Torch 2.2.2. Use the failing check as the next "
        "targeted compatibility patch instead of downgrading the whole stack."
    )
    raise SystemExit(1)

print("All forced macOS Intel compatibility checks passed.")
