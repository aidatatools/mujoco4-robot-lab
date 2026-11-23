"""第 0 課：檢查 MuJoCo 4 安裝了哪些元件。

  python scripts/check_install.py
"""

import importlib.metadata as md
import importlib.util
import os
import sys

print(f"Python {sys.version.split()[0]}")
for pkg in ["mujoco", "mujoco-core", "mujoco-studio", "mujoco-studio-wasm", "glfw", "pyopengl"]:
  try:
    print(f"  {pkg:20s} {md.version(pkg)}")
  except md.PackageNotFoundError:
    print(f"  {pkg:20s} (未安裝)")

import mujoco  # noqa: E402

print(f"\nmujoco 模組版本：{mujoco.__version__}")
checks = {
    "物理模擬 (mujoco-core)": True,
    "互動視窗 mujoco.viewer (需 glfw)": importlib.util.find_spec("glfw") is not None,
    "離線渲染 mujoco.Renderer (需 pyopengl)": importlib.util.find_spec("OpenGL") is not None,
    "MuJoCo Studio (mujoco-studio)": importlib.util.find_spec("mujoco.studio") is not None,
}
for k, v in checks.items():
  print(f"  {'✅' if v else '❌'} {k}")
print(f"\nMUJOCO_GL = {os.environ.get('MUJOCO_GL', '(未設定，有螢幕時用 glfw)')}")
