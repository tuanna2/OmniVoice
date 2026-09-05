#!/usr/bin/env python3
"""Patch Transformers 5.3 to recognize PyTorch 2.2.x as available.

This is intentionally scoped to the experimental macOS Intel runtime. Current
Transformers 5.3 contains Qwen3 and Higgs Audio V2 support needed by OmniVoice,
but its import gate reports torch<2.4 as unavailable before model code can even
be exercised. Intel macOS cannot install torch>=2.3 from official wheels, so we
lower only that availability gate and let the compatibility smoke tests expose
any real missing runtime APIs.
"""

from __future__ import annotations

import importlib.metadata
from pathlib import Path

EXPECTED_TRANSFORMERS = "5.3.0"
OLD_FLOOR = 'version.parse("2.4.0")'
NEW_FLOOR = 'version.parse("2.2.0")'
OLD_WARNING = "PyTorch >= 2.4 is required"
NEW_WARNING = "PyTorch >= 2.2 is required by the macOS Intel compatibility shim"


def main() -> None:
    version = importlib.metadata.version("transformers")
    if version != EXPECTED_TRANSFORMERS:
        raise SystemExit(
            f"Refusing to patch transformers {version}; expected {EXPECTED_TRANSFORMERS}."
        )

    dist = importlib.metadata.distribution("transformers")
    path = Path(dist.locate_file("transformers/utils/import_utils.py"))
    text = path.read_text(encoding="utf-8")

    # Idempotent: a previously patched environment should remain usable.
    if NEW_FLOOR in text and NEW_WARNING in text:
        print(f"Transformers torch availability gate already patched: {path}")
        return

    marker = "def is_torch_available() -> bool:"
    start = text.find(marker)
    if start < 0:
        raise SystemExit(f"Could not find is_torch_available() in {path}")
    end = text.find("\n\n@lru_cache", start + len(marker))
    if end < 0:
        raise SystemExit(f"Could not isolate is_torch_available() in {path}")

    before = text[start:end]
    if before.count(OLD_FLOOR) != 2 or OLD_WARNING not in before:
        raise SystemExit(
            "Transformers 5.3 torch availability gate no longer matches the "
            "expected source; refusing a broad/unsafe replacement."
        )

    after = before.replace(OLD_FLOOR, NEW_FLOOR).replace(OLD_WARNING, NEW_WARNING)
    path.write_text(text[:start] + after + text[end:], encoding="utf-8")
    print(f"Patched Transformers torch availability floor 2.4.0 -> 2.2.0: {path}")


if __name__ == "__main__":
    main()
