#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

ENV_NAME="${OMNIVOICE_ARM_ENV:-omnivoice}"

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "This installer is only for Apple Silicon macOS (Darwin arm64)." >&2
  echo "Detected: $(uname -s) $(uname -m)" >&2
  exit 1
fi
if ! command -v conda >/dev/null 2>&1; then
  echo "conda was not found in PATH." >&2
  exit 1
fi

conda env remove -n "$ENV_NAME" -y >/dev/null 2>&1 || true
conda env create -n "$ENV_NAME" -f environment.yml
conda run -n "$ENV_NAME" python -c 'import torch, transformers, omnivoice; print(f"torch={torch.__version__} transformers={transformers.__version__}")'

cat <<EOF
Apple Silicon OmniVoice environment is ready.
Activate it with:
  conda activate $ENV_NAME
EOF
