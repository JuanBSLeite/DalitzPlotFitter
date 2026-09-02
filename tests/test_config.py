import os
import subprocess
import sys


def _run_import(*, preallocate=None):
    env = os.environ.copy()
    if preallocate is None:
        env.pop("XLA_PYTHON_CLIENT_PREALLOCATE", None)
    else:
        env["XLA_PYTHON_CLIENT_PREALLOCATE"] = preallocate
    code = (
        "import os; import dalitzplotfitter; "
        "print(os.environ.get('XLA_PYTHON_CLIENT_PREALLOCATE'))"
    )
    return subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def test_jax_gpu_preallocation_is_disabled_by_default():
    assert _run_import() == "false"


def test_explicit_jax_gpu_preallocation_setting_is_preserved():
    assert _run_import(preallocate="true") == "true"
