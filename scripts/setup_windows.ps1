param(
  [ValidateSet("standard", "nvidia")]
  [string]$Profile = "standard",
  [string]$EnvName = $(if ($env:OMNIVOICE_WINDOWS_ENV) { $env:OMNIVOICE_WINDOWS_ENV } else { "omnivoice-windows" })
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
  throw "conda was not found in PATH"
}

try { conda env remove -n $EnvName -y 2>$null | Out-Null } catch {}
conda env create -n $EnvName -f environment-windows.yml

if ($Profile -eq "nvidia") {
  conda run -n $EnvName python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.8.0 torchaudio==2.8.0
} else {
  conda run -n $EnvName python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.8.0 torchaudio==2.8.0
}

conda run -n $EnvName python -m pip install -e .
conda run -n $EnvName python -c "import torch, transformers, omnivoice; print('torch=' + torch.__version__ + ' transformers=' + transformers.__version__)"
Write-Output "OmniVoice Windows profile '$Profile' is ready in conda env '$EnvName'."
