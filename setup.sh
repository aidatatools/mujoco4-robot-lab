#!/usr/bin/env bash
# MuJoCo 4.0.1 機器人模擬教學環境（Ubuntu 24.04 x86_64 / Python 3.12）
# 用法：bash setup.sh            （第一次會用 sudo 裝系統套件）
#       SKIP_APT=1 bash setup.sh （已裝過系統套件時）
set -euo pipefail
cd "$(dirname "$0")"
ROOT=$(pwd)

if [[ "${SKIP_APT:-0}" != "1" ]]; then
  sudo apt-get update
  sudo apt-get install -y git ffmpeg libegl1 libgl1 libglfw3
fi
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# 1) Python 3.12 虛擬環境 + MuJoCo 4.0.1 全套（core + studio + viewer）
uv venv -p 3.12 .venv
uv pip install -p .venv/bin/python "mujoco==4.0.1" "imageio[ffmpeg]" xacro

# 2) 取得兩台機器人的模型：版本固定在 third_party/versions.lock，確保每次環境一致
#    官方預訓練走路策略 BEST_WALK_ONNX_2.onnx 在 Open_Duck_Mini（v2 分支）
mkdir -p third_party
while read -r name url branch commit; do
  [[ -z "$name" || "$name" == \#* ]] && continue
  dir="third_party/$name"
  if [[ ! -d "$dir/.git" ]]; then
    git init -q "$dir"
    git -C "$dir" remote add origin "$url"
  fi
  if [[ "$(git -C "$dir" rev-parse HEAD 2>/dev/null)" != "$commit" ]]; then
    if [[ -n "$(git -C "$dir" status --porcelain --untracked-files=no 2>/dev/null)" ]]; then
      echo "❌ $dir 有未提交的修改，無法切換到鎖定版本 $commit（請先處理或刪除該目錄）" >&2
      exit 1
    fi
    echo "→ $name ($branch) @ ${commit:0:12}"
    git -C "$dir" fetch -q --depth 1 origin "$commit"
    git -C "$dir" -c advice.detachedHead=false checkout -q "$commit"
  fi
done < third_party/versions.lock

# 2b) Open Duck 專用環境：程式依賴舊版 JAX Playground，必須鎖 playground<1.0
#     版本全部鎖在 requirements-duck.txt（含 GPU 版 JAX、訓練要用的 TensorFlow / tf2onnx）
uv venv -p 3.12 .venv-duck
uv pip install -p .venv-duck/bin/python -r requirements-duck.txt

# 3) Open Duck：粗糙地形場景的 hfield 路徑在 MuJoCo 4 會變成 assets/assets/…，複製一份修正版
DUCK=third_party/Open_Duck_Playground/playground/open_duck_mini_v2/xmls
sed 's#file="assets/hfield.png"#file="hfield.png"#' "$DUCK/scene_rough_terrain_backlash.xml" \
  > "$DUCK/scene_rough_terrain_backlash_mj4.xml"

# 4) Mini Pupper 2：xacro → URDF（不需要裝 ROS；用真實質量 use_gazebo_hardware:=false）
DESC="$ROOT/third_party/mini_pupper_ros/mini_pupper_description"
mkdir -p models/mini_pupper_2 /tmp/mp_desc
rm -rf /tmp/mp_desc/mini_pupper_description && cp -r "$DESC" /tmp/mp_desc/
grep -rl '\$(find mini_pupper_description)' /tmp/mp_desc/mini_pupper_description/urdf \
  | xargs sed -i 's#\$(find mini_pupper_description)#/tmp/mp_desc/mini_pupper_description#g'
.venv/bin/xacro /tmp/mp_desc/mini_pupper_description/urdf/mini_pupper_2/mini_pupper_description.urdf.xacro \
  use_gazebo_hardware:=false > models/mini_pupper_2/mini_pupper_2.urdf

# 5) URDF → MJCF 場景
.venv/bin/python scripts/pupper_urdf_to_mjcf.py \
  --urdf models/mini_pupper_2/mini_pupper_2.urdf \
  --meshdir "$DESC/meshes/mini_pupper_2" \
  --out models/mini_pupper_2/scene.xml

# 6) 自我檢查：兩台都要站穩
.venv/bin/python scripts/check_install.py
MUJOCO_GL=egl .venv/bin/python scripts/stand_test.py --model "$DUCK/scene_flat_terrain.xml"
MUJOCO_GL=egl .venv/bin/python scripts/stand_test.py --model models/mini_pupper_2/scene.xml
# 走路自我檢查（無頭、固定路線）
MUJOCO_GL=egl .venv/bin/python scripts/pupper_trot.py --headless
MUJOCO_GL=egl .venv-duck/bin/python scripts/duck_walk.py --headless 2>&1 | grep -E "distance|Error"
echo "✅ 完成。Mini Pupper：source .venv/bin/activate    Open Duck：source .venv-duck/bin/activate"
