#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

ENV_NAME="${OMNIVOICE_INTEL_ENV:-omnivoice-intel}"
FULL_TEST=0

usage() {
  cat <<'EOF'
Usage: ./scripts/setup_macos_intel.sh [--full-test]

Creates or refreshes the experimental macOS Intel OmniVoice environment,
applies the Transformers 5.3 / PyTorch 2.2 compatibility shim, installs this
repository in editable mode, and runs the offline compatibility smoke test.

Options:
  --full-test   Also download k2-fsa/OmniVoice and run a tiny end-to-end CPU TTS.
  -h, --help    Show this help.

Environment:
  OMNIVOICE_INTEL_ENV   Conda environment name (default: omnivoice-intel)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --full-test)
      FULL_TEST=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "x86_64" ]]; then
  echo "This installer is only for Intel macOS (Darwin x86_64)." >&2
  echo "Detected: $(uname -s) $(uname -m)" >&2
  exit 1
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "conda was not found in PATH. Install Miniconda/Anaconda first." >&2
  exit 1
fi

echo "==> Intel macOS detected: $(sw_vers -productVersion) / $(uname -m)"
echo "==> Conda environment: $ENV_NAME"

if conda env list | awk 'NF && $1 !~ /^#/ {print $1}' | grep -Fxq "$ENV_NAME"; then
  echo "==> Refreshing existing environment"
  conda env update -n "$ENV_NAME" -f environment-intel.yml --prune
else
  echo "==> Creating environment"
  conda env create -n "$ENV_NAME" -f environment-intel.yml
fi

echo "==> Applying Transformers 5.3 compatibility shim for PyTorch 2.2.2"
conda run -n "$ENV_NAME" python scripts/patch_transformers_torch22.py

echo "==> Installing OmniVoice without re-resolving torch>=2.4 metadata"
conda run -n "$ENV_NAME" python -m pip install --no-deps -e .

echo "==> Running Intel compatibility smoke test"
conda run -n "$ENV_NAME" python scripts/check_macos_intel_compat.py

if [[ "$FULL_TEST" -eq 1 ]]; then
  echo "==> Running full checkpoint test (downloads k2-fsa/OmniVoice)"
  HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-300}" \
  HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-60}" \
  HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}" \
    conda run -n "$ENV_NAME" python scripts/check_macos_intel_full.py
fi

cat <<EOF

Intel OmniVoice environment is ready.

Activate it with:
  conda activate $ENV_NAME

Useful commands:
  omnivoice-server --host 127.0.0.1 --port 8003
  omnivoice-demo --ip 127.0.0.1 --port 8001

This Intel path is CPU-only and experimental. It is expected to be much slower
than Apple Silicon MPS or CUDA for normal-length synthesis.
EOF
