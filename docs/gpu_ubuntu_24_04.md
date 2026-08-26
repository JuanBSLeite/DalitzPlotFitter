# NVIDIA GPU testing on Ubuntu 24.04.4 LTS

This guide defines the reference GPU test environment for DalitzPlotFitter:

```text
Operating system: Ubuntu 24.04.4 LTS
Python:           3.12.3
Backend:          JAX
Accelerator:      NVIDIA GPU via CUDA
```

DalitzPlotFitter has no separate GPU code path. The numerical backend is JAX, so the same fitter code automatically runs on an NVIDIA GPU when a CUDA-enabled JAX installation detects one.

## 1. Verify the operating system and Python version

Check Ubuntu:

```bash
lsb_release -a
```

The reference system should report Ubuntu 24.04.4 LTS.

Check Python:

```bash
python3.12 --version
```

The reference version used for this test is:

```text
Python 3.12.3
```

You can enforce this check before creating the environment:

```bash
python3.12 -c 'import sys; assert sys.version_info[:3] == (3, 12, 3), sys.version; print(sys.version)'
```

If that assertion fails, the machine is not using the exact reference Python version described in this guide.

## 2. Verify that Ubuntu sees the NVIDIA GPU

```bash
lspci | grep -i nvidia
```

If the NVIDIA driver is already installed, check it with:

```bash
nvidia-smi
```

If `nvidia-smi` reports the GPU name and driver version correctly, continue to the Python environment setup.

## 3. Install the NVIDIA driver if necessary

Ubuntu recommends installing NVIDIA drivers through its packaged `ubuntu-drivers` utility rather than downloading a driver installer directly from NVIDIA.

```bash
sudo apt update
sudo apt upgrade
sudo apt install -y ubuntu-drivers-common
sudo ubuntu-drivers list
sudo ubuntu-drivers install
sudo reboot
```

After rebooting:

```bash
nvidia-smi
```

Current JAX requirements on Linux are:

- CUDA 13 wheels: NVIDIA driver >= 580 and GPU compute capability >= 7.5;
- CUDA 12 wheels: NVIDIA driver >= 525 and GPU compute capability >= 5.2.

For current hardware, CUDA 13 is the preferred JAX installation when the driver and GPU support it.

Official references:

- JAX installation: https://docs.jax.dev/en/latest/installation.html
- Ubuntu NVIDIA driver installation: https://ubuntu.com/desktop/docs/en/latest/how-to/graphics/install-nvidia-drivers/

## 4. Create the Python 3.12.3 environment

From the DalitzPlotFitter repository:

```bash
cd DalitzPlotFitter
python3.12 -m venv .venv
source .venv/bin/activate
```

Verify that the virtual environment is using the expected interpreter:

```bash
python --version
```

Expected:

```text
Python 3.12.3
```

For an exact automated check:

```bash
python -c 'import sys; assert sys.version_info[:3] == (3, 12, 3), sys.version; print("Python:", sys.version)'
```

Then install the project:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## 5. Install JAX with NVIDIA CUDA support

The JAX project recommends using the CUDA and cuDNN libraries distributed through its pip dependencies. With this method, a complete system-wide CUDA Toolkit installation is not required.

### CUDA 13 — recommended for recent GPUs

Use this when the NVIDIA driver is version 580 or newer and the GPU has compute capability 7.5 or newer:

```bash
python -m pip install --upgrade "jax[cuda13]"
```

### CUDA 12 — compatibility option

Use this for a supported older GPU or a Linux driver in the 525–579 range:

```bash
python -m pip install --upgrade "jax[cuda12]"
```

When using the pip-provided CUDA libraries, avoid pointing `LD_LIBRARY_PATH` at an unrelated CUDA installation because it can override the libraries selected by JAX and create version conflicts.

Check whether this variable is set:

```bash
echo "$LD_LIBRARY_PATH"
```

For the pip-managed CUDA route, an empty value is generally preferable unless the machine has another explicit requirement.

## 6. Verify that JAX is actually using the GPU

Run:

```bash
python - <<'PY'
import sys
import jax

print("Python:", sys.version)
print("JAX:", jax.__version__)
print("Default backend:", jax.default_backend())
print("Devices:", jax.devices())

assert sys.version_info[:3] == (3, 12, 3), sys.version
assert jax.default_backend() == "gpu", "JAX is not using the NVIDIA GPU"
assert any(device.platform == "gpu" for device in jax.devices())
PY
```

A successful setup should contain output similar to:

```text
Python: 3.12.3 ...
Default backend: gpu
Devices: [CudaDevice(id=0)]
```

The exact device description can vary between JAX releases and GPU models.

## 7. Run a small JAX GPU calculation

Before running the fitter, test an actual computation:

```bash
python - <<'PY'
import jax
import jax.numpy as jnp

x = jnp.ones((4096, 4096))
y = jax.jit(lambda a: a @ a)(x)
y.block_until_ready()

print("Backend:", jax.default_backend())
print("Result device:", y.device)
print("Result:", float(y[0, 0]))
PY
```

The backend must be `gpu` and the result array must reside on a CUDA device.

## 8. Run the DalitzPlotFitter test suite on the GPU

With the same environment active:

```bash
python -c 'import jax; assert jax.default_backend() == "gpu"; print(jax.devices())' && pytest
```

This first checks that a GPU is visible and only then starts the test suite.

For the fitter closure test specifically:

```bash
python -c 'import jax; assert jax.default_backend() == "gpu"; print(jax.devices())' \
  && pytest -v tests/test_fit_closure.py
```

For the `pi- pi+ pi+` kinematic/amplitude validation:

```bash
pytest -v tests/test_pi_minus_first_ordering.py
```

## 9. Run the closure notebook on the GPU

Start Jupyter from the same virtual environment:

```bash
jupyter lab notebooks/01_dplus_fit_closure.ipynb
```

At the beginning of the notebook, verify:

```python
import sys
import jax

print(sys.version)
print(jax.default_backend())
print(jax.devices())

assert sys.version_info[:3] == (3, 12, 3)
assert jax.default_backend() == "gpu"
```

The closure notebook then runs the same `D+ -> pi- pi+ pi+` model with the GPU-backed JAX implementation.

## 10. Monitor GPU use during a fit

In a second terminal:

```bash
watch -n 1 nvidia-smi
```

During JAX compilation and numerical evaluation, the Python process should appear in the GPU process list and GPU memory usage should increase.

Note that a coefficient-only fit deliberately uses the cached component amplitudes and cached normalization matrix. After the initial JAX calculations, each likelihood evaluation is comparatively small, so GPU utilization does not need to remain close to 100% throughout the entire minimization.

## Multi-GPU machines

To expose only one GPU to JAX:

```bash
export JAX_CUDA_VISIBLE_DEVICES=0
```

Then verify:

```bash
python -c 'import jax; print(jax.devices())'
```

## Shared GPU machines

JAX normally preallocates a large fraction of GPU memory. For interactive testing on a shared machine, preallocation can be disabled before starting Python or Jupyter:

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
```

This can make GPU sharing easier. For dedicated performance benchmarks, the default allocator can provide better and more predictable performance.

## Minimal reference test sequence

For an already configured Ubuntu 24.04.4 LTS machine with NVIDIA drivers installed, the shortest reference sequence is:

```bash
cd DalitzPlotFitter

python3.12 -c 'import sys; assert sys.version_info[:3] == (3, 12, 3), sys.version'

python3.12 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pip install --upgrade "jax[cuda13]"

python - <<'PY'
import sys
import jax
assert sys.version_info[:3] == (3, 12, 3), sys.version
assert jax.default_backend() == "gpu"
print("Python:", sys.version)
print("JAX devices:", jax.devices())
PY

pytest -v tests/test_fit_closure.py
```

Use `jax[cuda12]` instead of `jax[cuda13]` when required by the installed NVIDIA driver or GPU generation.
