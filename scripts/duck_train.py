"""Train an Open Duck Mini v2 walking policy with Open_Duck_Playground (PPO, JAX).

Needs the .venv-duck environment (see requirements-duck.txt).

This is a thin launcher around the upstream script
  third_party/Open_Duck_Playground/playground/open_duck_mini_v2/runner.py
that
  1. loads jax_compat.py so brax 0.14 works on JAX >= 0.10, and
  2. runs from inside the Open_Duck_Playground folder, because the upstream
     runner always writes ONNX.onnx and .tmp/jax_cache to the current folder.

All arguments are passed through to the upstream runner:
  python scripts/duck_train.py                          # full run (150M steps)
  python scripts/duck_train.py --num_timesteps 2000000  # quick smoke test
  python scripts/duck_train.py --help

Checkpoints and .onnx policies go to
third_party/Open_Duck_Playground/checkpoints/ unless --output_dir is an
absolute path.
"""

import os
import pathlib
import runpy
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
DUCK_REPO = ROOT / "third_party" / "Open_Duck_Playground"

sys.path.insert(0, str(DUCK_REPO))
import jax_compat  # noqa: E402,F401  (scripts/ is on sys.path as the script's folder)

# TensorFlow is only used for the small ONNX export inside the training loop.
# Keep it on the CPU: JAX already preallocated most of the GPU memory, so TF
# cannot create a cuBLAS handle ("No blas support for stream"), and TF has no
# prebuilt kernels for newer GPUs such as compute capability 12.0 anyway.
# This must run before TF touches the GPU.
import tensorflow as tf  # noqa: E402

tf.config.set_visible_devices([], "GPU")

os.chdir(DUCK_REPO)
sys.argv[0] = "playground/open_duck_mini_v2/runner.py"
runpy.run_module("playground.open_duck_mini_v2.runner", run_name="__main__", alter_sys=True)
