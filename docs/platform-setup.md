# Platform setup

This fork keeps the upstream PyTorch 2.8 path for Apple Silicon and Windows, and adds a separate Intel macOS compatibility path.

## macOS Intel

```bash
./scripts/setup_macos_intel.sh
```

The Intel environment uses Python 3.11, PyTorch/Torchaudio 2.2.2 and the narrow Transformers 5.3 compatibility shim in `scripts/patch_transformers_torch22.py`.

## macOS Apple Silicon

```bash
./scripts/setup_macos_arm64.sh
```

This uses the normal PyTorch 2.8 / MPS path.

## Windows x64 standard CPU

```powershell
.\scripts\setup_windows_standard.ps1
```

## Windows x64 NVIDIA

```powershell
.\scripts\setup_windows_nvidia.ps1
```

The NVIDIA setup installs the CUDA 12.8 PyTorch 2.8 wheels.
