#!/usr/bin/env python3
"""Patch Transformers 5.3 for the experimental PyTorch 2.2 Intel runtime.

Transformers 5.3 contains the Qwen3 and Higgs Audio V2 implementations current
OmniVoice needs, but it deliberately disables torch<2.4 and also assumes a few
APIs/dtypes added after the last macOS x86_64 PyTorch release. This script makes
only narrow, source-checked compatibility edits so the real model paths can be
exercised on torch 2.2.2.
"""

from __future__ import annotations

import importlib.metadata
from pathlib import Path

EXPECTED_TRANSFORMERS = "5.3.0"


def _distribution_file(dist, relative: str) -> Path:
    return Path(dist.locate_file(relative))


def _patch_torch_availability(dist) -> None:
    path = _distribution_file(dist, "transformers/utils/import_utils.py")
    text = path.read_text(encoding="utf-8")
    marker = "def is_torch_available() -> bool:"
    start = text.find(marker)
    if start < 0:
        raise SystemExit(f"Could not find is_torch_available() in {path}")
    end = text.find("\n\n@lru_cache", start + len(marker))
    if end < 0:
        raise SystemExit(f"Could not isolate is_torch_available() in {path}")

    before = text[start:end]
    old_floor = 'version.parse("2.4.0")'
    new_floor = 'version.parse("2.2.0")'
    old_warning = "PyTorch >= 2.4 is required"
    new_warning = "PyTorch >= 2.2 is required by the macOS Intel compatibility shim"

    if new_floor in before and new_warning in before:
        print(f"Torch availability gate already patched: {path}")
        return
    if before.count(old_floor) != 2 or old_warning not in before:
        raise SystemExit(
            "Transformers 5.3 torch availability gate no longer matches the "
            "expected source; refusing a broad replacement."
        )

    after = before.replace(old_floor, new_floor).replace(old_warning, new_warning)
    path.write_text(text[:start] + after + text[end:], encoding="utf-8")
    print(f"Patched Transformers torch availability floor 2.4.0 -> 2.2.0: {path}")


def _patch_modeling_utils(dist) -> None:
    path = _distribution_file(dist, "transformers/modeling_utils.py")
    text = path.read_text(encoding="utf-8")
    original = text

    # torch.get_default_device() is newer than the Intel-supported torch line.
    old = "    default_device = torch.get_default_device()"
    new = (
        '    default_device = (\n'
        '        torch.get_default_device() if hasattr(torch, "get_default_device") else torch.device("cpu")\n'
        '    )'
    )
    if old in text:
        text = text.replace(old, new, 1)
    elif new not in text:
        raise SystemExit(f"Unexpected get_default_device source in {path}")

    # New unsigned/float8 dtype attributes are used only to decode matching
    # safetensors dtypes. Do not alias them to a wrong dtype on torch 2.2;
    # simply omit mappings for attributes this torch build does not expose.
    dtype_replacements = {
        '    "U16": torch.uint16,': '    **({"U16": torch.uint16} if hasattr(torch, "uint16") else {}),',
        '    "U32": torch.uint32,': '    **({"U32": torch.uint32} if hasattr(torch, "uint32") else {}),',
        '    "U64": torch.uint64,': '    **({"U64": torch.uint64} if hasattr(torch, "uint64") else {}),',
        '    "F8_E4M3": torch.float8_e4m3fn,': '    **({"F8_E4M3": torch.float8_e4m3fn} if hasattr(torch, "float8_e4m3fn") else {}),',
        '    "F8_E5M2": torch.float8_e5m2,': '    **({"F8_E5M2": torch.float8_e5m2} if hasattr(torch, "float8_e5m2") else {}),',
    }
    for old_line, new_line in dtype_replacements.items():
        if old_line in text:
            text = text.replace(old_line, new_line, 1)
        elif new_line not in text:
            raise SystemExit(f"Unexpected dtype mapping source in {path}: {old_line.strip()}")

    if text == original:
        print(f"Transformers modeling_utils compatibility already patched: {path}")
        return

    path.write_text(text, encoding="utf-8")
    print(f"Patched Transformers modeling_utils for torch 2.2: {path}")


def main() -> None:
    version = importlib.metadata.version("transformers")
    if version != EXPECTED_TRANSFORMERS:
        raise SystemExit(
            f"Refusing to patch transformers {version}; expected {EXPECTED_TRANSFORMERS}."
        )

    dist = importlib.metadata.distribution("transformers")
    _patch_torch_availability(dist)
    _patch_modeling_utils(dist)


if __name__ == "__main__":
    main()
