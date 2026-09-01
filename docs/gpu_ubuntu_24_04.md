# NVIDIA GPU testing on Ubuntu 24.04.4 LTS / WSL2

This guide defines the reference GPU test environment for DalitzPlotFitter:

```text
Host:             Windows + WSL2
Guest OS:         Ubuntu 24.04.4 LTS
Python:           3.12.3
Backend:          JAX
Accelerator:      NVIDIA GPU via CUDA
Reference GPU:    GeForce RTX 3050 Ti Laptop GPU (4 GB, compute capability 8.6)
```

DalitzPlotFitter has no separate GPU code path. The numerical backend is JAX, so the same fitter code automatically runs on an NVIDIA GPU when a CUDA-enabled JAX installation detects one.

## Important WSL2 rule

When using WSL2, install the NVIDIA display driver on the **Windows host only**. Do **not** install a Linux NVIDIA display driver inside the Ubuntu WSL distribution. WSL exposes the Windows NVIDIA driver to Linux applications.

The CUDA Toolkit itself is also not required when using the CUDA libraries distributed through the JAX pip installation.

Official references:

- JAX installation: https://docs.jax.dev/en/latest/installation.html
- JAX GPU memory allocation: https://docs.jax.dev/en/latest/gpu_memory_allocation.html
- NVIDIA CUDA on WSL: https://docs.nvidia.com/cuda/wsl-user-guide/index.html
- Microsoft WSL configuration: https://learn.microsoft.com/windows/wsl/wsl-config

## 1. Verify WSL2, Ubuntu and Python

From Windows PowerShell:

```powershell
wsl --status
wsl --version
```

From the Ubuntu WSL shell:

```bash
lsb_release -a
python3.12 --version
```

The reference environment is:

```text
Ubuntu 24.04.4 LTS
Python 3.12.3
```

For an exact Python check:

```bash
python3.12 -c 'import sys; assert sys.version_info[:3] == (3, 12, 3), sys.version; print(sys.version)'
```

## 2. Verify NVIDIA GPU access from WSL

Run inside Ubuntu WSL:

```bash
nvidia-smi
```

A working WSL CUDA setup should show the NVIDIA GPU and the Windows-host driver version.

Do not run `ubuntu-drivers install` inside WSL. If the driver needs updating, update the NVIDIA Windows driver instead.

It is also useful to update WSL itself from PowerShell:

```powershell
wsl --update
```

## 3. RTX 3050 Ti compatibility

The GeForce RTX 3050 Ti Laptop GPU is an Ampere GPU with compute capability 8.6 and typically 4 GB of GDDR6 memory. It is compatible with JAX CUDA 12 and also satisfies the GPU architecture requirement for CUDA 13.

For this project on WSL2, CUDA 12 is the conservative reference configuration because it provides broad driver compatibility and is sufficient for the fitter.

## 4. Create the Python 3.12.3 environment

From the DalitzPlotFitter repository:

```bash
cd DalitzPlotFitter
python3.12 -m venv .venv
source .venv/bin/activate
```

Verify:

```bash
python --version
```

Expected:

```text
Python 3.12.3
```

Install the project:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## 5. Install exactly one JAX CUDA plugin set

Do not install CUDA 12 and CUDA 13 JAX plugins in the same virtual environment.

For the RTX 3050 Ti / WSL2 reference configuration:

```bash
python -m pip install --upgrade "jax[cuda12]"
```

Use CUDA 13 only when the Windows NVIDIA driver is new enough and there is a specific reason to move to it:

```bash
python -m pip install --upgrade "jax[cuda13]"
```

## 6. Fix duplicate PJRT CUDA registration

A possible installation error is:

```text
ALREADY_EXISTS: PJRT_Api already exists for device type cuda
```

If `jax.devices()` still prints `[CudaDevice(id=0)]`, the GPU is visible; the error usually indicates a duplicate/conflicting CUDA plugin installation.

Inspect the environment:

```bash
python -m pip list | grep -Ei '^(jax|jaxlib|jax-cuda|nvidia-.*cu1[23])'
```

The cleanest development fix is to recreate the virtual environment and install only one CUDA family:

```bash
deactivate 2>/dev/null || true
rm -rf .venv

python3.12 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pip install --no-cache-dir --upgrade "jax[cuda12]"
```

Then verify:

```bash
python - <<'PY'
import jax
import jaxlib
print("JAX:", jax.__version__)
print("JAXLIB:", jaxlib.__version__)
print("Backend:", jax.default_backend())
print("Devices:", jax.devices())
assert jax.default_backend() == "gpu"
PY
```

## 7. RTX 3050 Ti / WSL2 memory configuration

The RTX 3050 Ti Laptop GPU normally has 4 GB of VRAM. JAX preallocates a large fraction of GPU memory by default, which is unnecessarily aggressive for interactive work on this GPU.

Before starting VS Code, Jupyter, or Python, use:

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
```

Then launch VS Code or Jupyter from the same shell so the kernel inherits the variable:

```bash
code .
```

or:

```bash
jupyter lab
```

An alternative is to keep preallocation enabled but reduce the fraction, for example:

```bash
export XLA_CLIENT_MEM_FRACTION=0.45
```

Do not rely on setting these variables after JAX has already been imported in a running notebook kernel. Restart the kernel/process after changing them.

For debugging with the smallest possible GPU footprint, JAX also supports:

```bash
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
```

This allocator is slower and is intended mainly for debugging memory problems.

## 8. WSL pinned-memory limitation

WSL2 has a documented CUDA limitation: the amount of pinned system memory available to CUDA applications is limited.

A crash such as:

```text
Failed to allocate 2.00MiB of pinned host memory
CUDA_ERROR_OUT_OF_MEMORY
```

is therefore not necessarily a 4 GB VRAM exhaustion. It can also be the WSL CUDA pinned-host-memory limit or host-memory pressure.

Before retrying, stop stale WSL/Jupyter/Python processes. From Windows PowerShell, the strongest reset is:

```powershell
wsl --shutdown
```

Then reopen Ubuntu WSL, activate the environment, export the JAX memory setting, and start the notebook again:

```bash
cd ~/work/DalitzPlotFitter
source .venv/bin/activate
export XLA_PYTHON_CLIENT_PREALLOCATE=false
code .
```

Check host memory inside WSL:

```bash
free -h
```

Check GPU memory:

```bash
nvidia-smi
```

If WSL is receiving too little RAM, Windows can assign more through `%UserProfile%\.wslconfig`. For example, on a machine with enough physical RAM:

```ini
[wsl2]
memory=8GB
swap=8GB
```

Apply changes with:

```powershell
wsl --shutdown
```

The exact WSL memory value should leave sufficient RAM available for Windows itself.

## 9. Verify JAX GPU detection

```bash
python - <<'PY'
import sys
import jax
import jaxlib

print("Python:", sys.version)
print("JAX:", jax.__version__)
print("JAXLIB:", jaxlib.__version__)
print("Default backend:", jax.default_backend())
print("Devices:", jax.devices())

assert sys.version_info[:3] == (3, 12, 3), sys.version
assert jax.default_backend() == "gpu"
assert any(device.platform == "gpu" for device in jax.devices())
PY
```

Expected output contains something similar to:

```text
Default backend: gpu
Devices: [CudaDevice(id=0)]
```

## 10. Run a small JAX GPU calculation

For the 4 GB RTX 3050 Ti, use a moderate matrix size for the smoke test:

```bash
python - <<'PY'
import jax
import jax.numpy as jnp

x = jnp.ones((2048, 2048))
y = jax.jit(lambda a: a @ a)(x)
y.block_until_ready()

print("Backend:", jax.default_backend())
print("Result device:", y.device)
print("Result:", float(y[0, 0]))
PY
```

## 11. Run DalitzPlotFitter tests on the GPU

```bash
python -c 'import jax; assert jax.default_backend() == "gpu"; print(jax.devices())' && pytest
```

For the closure test specifically:

```bash
python -c 'import jax; assert jax.default_backend() == "gpu"; print(jax.devices())' \
  && pytest -v tests/test_fit_closure.py
```

For the `pi- pi+ pi+` kinematic/amplitude validation:

```bash
pytest -v tests/test_pi_minus_first_ordering.py
```

## 12. Run the closure notebook on the GPU

Always launch the notebook process after setting the memory configuration:

```bash
source .venv/bin/activate
export XLA_PYTHON_CLIENT_PREALLOCATE=false
jupyter lab notebooks/01_e791_toy_fit.ipynb
```

or launch VS Code from that shell:

```bash
source .venv/bin/activate
export XLA_PYTHON_CLIENT_PREALLOCATE=false
code .
```

At the beginning of the notebook:

```python
import sys
import jax

print(sys.version)
print(jax.default_backend())
print(jax.devices())

assert sys.version_info[:3] == (3, 12, 3)
assert jax.default_backend() == "gpu"
```

## 13. Monitor GPU use

From another WSL terminal:

```bash
watch -n 1 nvidia-smi
```

A coefficient-only fit deliberately uses cached component amplitudes and a cached normalization matrix. After the initial JAX calculations, individual likelihood evaluations are small, so GPU utilization does not need to remain close to 100% throughout the minimization.

## Minimal RTX 3050 Ti / WSL2 reference sequence

```bash
cd ~/work/DalitzPlotFitter
source .venv/bin/activate

export XLA_PYTHON_CLIENT_PREALLOCATE=false

python - <<'PY'
import sys
import jax
assert sys.version_info[:3] == (3, 12, 3), sys.version
assert jax.default_backend() == "gpu"
print("Python:", sys.version)
print("Devices:", jax.devices())
PY

pytest -v tests/test_pi_minus_first_ordering.py
pytest -v tests/test_fit_closure.py
```

If a notebook kernel previously crashed with a CUDA pinned-host-memory error, run `wsl --shutdown` from Windows PowerShell before retrying this sequence.
