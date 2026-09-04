#!/usr/bin/env python3
"""Smoke-test the experimental macOS Intel dependency set."""

import platform
import sys

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
checks.append(
    check(
        "PyTorch 2.2.2 import",
        lambda: (_ for _ in ()).throw(
            RuntimeError(f"expected torch 2.2.2, got {torch.__version__}")
        )
        if not torch.__version__.startswith("2.2.2")
        else None,
    )
)


def check_qwen3():
    from transformers import AutoConfig

    AutoConfig.for_model("qwen3")


checks.append(check("Transformers Qwen3 support", check_qwen3))


def check_higgs():
    from transformers import HiggsAudioV2TokenizerModel  # noqa: F401


checks.append(check("HiggsAudioV2TokenizerModel support", check_higgs))


def check_omnivoice_import():
    from omnivoice.models.omnivoice import OmniVoice  # noqa: F401


checks.append(check("OmniVoice core import", check_omnivoice_import))

if not all(checks):
    print(
        "\nThe Intel dependency set installs, but stock Transformers 4.43.4 "
        "is not sufficient for current OmniVoice. The failed capability checks "
        "show what must be backported or replaced next."
    )
    raise SystemExit(1)

print("All macOS Intel compatibility checks passed.")
