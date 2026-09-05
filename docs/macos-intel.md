# Experimental macOS Intel runtime

OmniVoice's normal dependency set requires PyTorch 2.4 or newer, but official
macOS x86_64 PyTorch wheels stop at 2.2.2. This branch contains a compatibility
runtime that keeps the current OmniVoice userspace (`transformers==5.3.0`,
Qwen3, and Higgs Audio V2) while running on `torch==2.2.2` and
`torchaudio==2.2.2`.

## Verified configuration

The GitHub Actions `macos-15-intel` runner has verified all of the following on
real x86_64 macOS:

- Python 3.11
- PyTorch 2.2.2 / Torchaudio 2.2.2
- Transformers 5.3.0 with the narrow compatibility shim
- Qwen3 forward pass
- Higgs Audio V2 encode, decode, and forward pass
- OmniVoice model construction and VoiceClonePrompt save/load
- Full `k2-fsa/OmniVoice` checkpoint loading on CPU
- End-to-end OmniVoice generation through Qwen3 and the real Higgs tokenizer

The full integration check generated a finite 1-D waveform from the real model
on the Intel runner. This validates compatibility, not synthesis speed or audio
quality on Intel CPUs.

## One-command setup

From this repository branch:

```bash
./scripts/setup_macos_intel.sh
```

The script creates (or refreshes) the `omnivoice-intel` Conda environment,
applies the Transformers compatibility shim, installs OmniVoice in editable
mode without re-resolving the normal `torch>=2.4` project metadata, and runs the
offline smoke test.

To also download the real checkpoint and repeat the end-to-end test:

```bash
./scripts/setup_macos_intel.sh --full-test
```

Use another environment name if desired:

```bash
OMNIVOICE_INTEL_ENV=omnivoice-intel-test ./scripts/setup_macos_intel.sh
```

After setup:

```bash
conda activate omnivoice-intel
omnivoice-server --host 127.0.0.1 --port 8003
```

## What the compatibility shim changes

`scripts/patch_transformers_torch22.py` only accepts exactly Transformers
5.3.0 and makes source-checked edits to the installed package. It lowers the
Transformers PyTorch availability gate from 2.4 to 2.2 and adds fallbacks for a
small set of post-2.2 APIs used by current Transformers, including default
device handling and the autocast query used by Qwen3. Unsupported newer
safetensors dtypes are omitted rather than aliased to incorrect dtypes.

The shim deliberately does not replace Qwen3 or Higgs with older
implementations and does not monkey-patch PyTorch globally.

## Intel-specific dependency pins

`environment-intel.yml` also pins:

- `numpy<2`
- `numba==0.60.0`
- `llvmlite==0.43.0`
- `librosa==0.11.0`

The Numba/llvmlite pair is intentional: newer llvmlite releases can lack a
CPython 3.11 macOS x86_64 wheel and fall back to an LLVM source build.

## Limitations

- Intel Mac inference is CPU-only; there is no Apple MPS acceleration on Intel.
- Normal-length TTS will be substantially slower than Apple Silicon MPS or an
  NVIDIA CUDA worker.
- This is a compatibility path pinned to Transformers 5.3.0 and PyTorch 2.2.2.
  Re-run the smoke/full tests before changing either version.
- The normal `pyproject.toml` remains aligned with upstream requirements, so
  this path intentionally installs the local package with `--no-deps` after
  the Intel environment has been resolved.
