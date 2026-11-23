# MuJoCo 4.0.1 機器人模擬實驗：Open Duck Mini v2 + Mini Pupper 2

環境：Ubuntu 24.04 x86_64、Python 3.12、MuJoCo 4.0.1（RTX 5090 筆電已驗證可用 EGL 離線渲染）

---

## 0. MuJoCo 4 的套件拆分（先講清楚，學員最常卡在這）

從 4.0.0 開始，`mujoco` 變成「總包」，底下拆成幾個套件：

| 安裝指令 | 裝了什麼 | 能做什麼 | 適合 |
|---|---|---|---|
| `pip install mujoco-core` | C 函式庫 + Python 綁定，只依賴 numpy | **只能模擬**，沒有視窗、沒有渲染 | 訓練伺服器、headless RL、CI |
| `pip install "mujoco-core[rendering]"` | 上面 + glfw + pyopengl | + `mujoco.viewer` 互動視窗、`mujoco.Renderer` 離線渲染 | 一般開發、錄影 |
| `pip install mujoco-studio` | core + Studio + 網頁檢視器 | + `mujoco-studio` 新版 GUI、瀏覽器檢視 | 教學展示、遠端看模擬 |
| `pip install mujoco` | **全部**（core + studio + glfw + pyopengl） | 以上全部 | **課堂預設用這個** |

不管裝哪個，程式裡都是 `import mujoco`，寫法不變。

實測（`python scripts/check_install.py`）：

```
只裝 mujoco-core：   ✅ 模擬   ❌ viewer   ❌ Renderer   ❌ Studio
裝 mujoco：          ✅ 模擬   ✅ viewer   ✅ Renderer   ✅ Studio
```

> 從 3.x 升級時，pip 可能因為檔案搬到 `mujoco-core` 而裝壞，先 `pip uninstall mujoco` 再裝，或用全新的虛擬環境。

---

## 1. 一鍵安裝

```bash
git clone <本專案> mujoco4-robot-lab && cd mujoco4-robot-lab
bash setup.sh
source .venv/bin/activate
```

`setup.sh` 會做這些事：

1. 建兩個 Python 3.12 虛擬環境：
   - `.venv`：`mujoco==4.0.1`，給第 2～4 節和 Mini Pupper 用
   - `.venv-duck`：Open Duck 專用（JAX + 舊版 Playground），套件版本鎖在 `requirements-duck.txt`，見第 5 節
2. clone `Open_Duck_Playground`、`Open_Duck_Mini`（v2 分支，預訓練策略）和 `mini_pupper_ros`（ros2-jazzy 分支），並切到 `third_party/versions.lock` 固定的 commit，確保每個人拿到相同的上游版本。要升級上游：改 lock 檔的 commit → 重跑 `setup.sh` → 自我檢查通過再提交
3. 修正 Open Duck 粗糙地形場景的 hfield 路徑（另存成 `*_mj4.xml`）
4. 把 Mini Pupper 2 的 xacro 展開成 URDF（**不需要裝 ROS**）
5. URDF → MJCF 場景
6. 自我檢查：兩台機器人都要「站穩 ✅」

---

## 2. 三種看模擬的方式

```bash
DUCK=third_party/Open_Duck_Playground/playground/open_duck_mini_v2/xmls/scene_flat_terrain.xml
PUPPER=models/mini_pupper_2/scene.xml
```

**(a) 經典互動視窗**（需要 `glfw`）

```bash
python -m mujoco.viewer --mjcf=$DUCK
```

**(b) MuJoCo Studio**（4.0 新 GUI）

```bash
mujoco-studio $PUPPER
mujoco-studio $PUPPER --hot_reload=model     # 改 XML 存檔就自動重載，教學很好用
```

**(c) Studio 網頁檢視器**（學員用瀏覽器看，不用裝任何東西）

```bash
mujoco-studio --gfx=web --port 8080 $PUPPER
# 瀏覽器打開 http://<5090 筆電 IP>:8080
```

> ⚠️ 網頁檢視器**沒有帳號密碼也沒有加密**，只能在區網或 SSH tunnel 裡用，不要對外開 port。

---

## 3. 實驗一：站立測試（PD 位置控制）

```bash
python scripts/stand_test.py --model $DUCK
python scripts/stand_test.py --model $PUPPER
```

預期輸出：

```
Open Duck  ：機身高度 0.150 → 0.160 m，最大傾角 6.8°  → 站穩 ✅
Mini Pupper：機身高度 0.120 → 0.075 m，最大傾角 0.5°  → 站穩 ✅
```

加選項：

```bash
python scripts/stand_test.py --model $PUPPER --viewer                        # 邊跑邊看
MUJOCO_GL=egl python scripts/stand_test.py --model $PUPPER --video pupper.mp4  # 無頭錄影
python scripts/stand_test.py --model $PUPPER --push 15                        # 第 1 秒側推 15 N → 會倒
```

**課堂討論**：把推力從 2 N 慢慢加大，找出每台機器人倒下的臨界值。
- 純 PD 站立撐不住多大的推力。
- 這就是為什麼要用 RL 學平衡策略。

---

## 4. 實驗二：Mini Pupper 2 的 URDF → MJCF（`scripts/pupper_urdf_to_mjcf.py`）

這一段是 sim2real 最重要的觀念課。轉換時做了 4 件事：

1. **用真實質量展開 xacro**：`use_gazebo_hardware:=false`。預設值會把質量放大 30 倍、慣量放大 100 倍，那是為了讓 Gazebo ODE 不爆掉的權宜做法，**拿來做 RL 一定會出錯**。
2. **加 free joint**：URDF 的根連桿預設焊死在世界座標上，要自己加浮動基座。
3. **補致動器**：URDF 沒有致動器，要用 MjSpec 幫 12 個關節各加一個位置控制器（kp、kv、扭力上限）。
4. **伺服參數**：URDF 裡的 `damping=5.0` 和 `effort=10` 都是佔位值。腳本裡的 `SERVO` 字典是**教學起始值**，做 sim2real 前一定要拿實機做系統辨識。

### MuJoCo 4 的 MjSpec API 變更（舊範例會報錯）

| 3.x 寫法 | 4.0 寫法 |
|---|---|
| `joint.damping = 0.01` | `joint.damping = [0.01, 0, 0]`（變成多項式係數） |
| `joint.stiffness = 2.0` | `joint.stiffness = [2.0, 0, 0]` |
| `add_light(directional=True)` | `add_light(type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)` |

### Open Duck 在 MuJoCo 4 的相容性（實測）

- `scene_flat_terrain.xml`、`scene_flat_terrain_backlash.xml`：直接可用。
- `scene_rough_terrain_backlash.xml`：hfield 路徑要修正（`setup.sh` 已另存成 `_mj4.xml`）。
- 積分器是 Euler，不受 4.0「RK4 + free joint 行為改變」影響。

---

## 5. 進階：用 Open Duck Playground 訓練走路策略（GPU）

### 為什麼要另外一個環境、還要鎖版本

- `Open_Duck_Playground` 的 `pyproject.toml` 寫 `playground>=0.0.3`，現在會裝到 **Playground 1.0**。1.0 已改成 PyTorch + MuJoCo Warp，**原專案的 JAX 程式會壞掉**，必須鎖 `playground==0.0.5`。
- 訓練用的 brax 0.14.2（目前最新版）還在呼叫 JAX 0.10 已經移除的 `jax.device_put_replicated`。新版 Flax 又要求 JAX ≥ 0.11.1，沒辦法降版 JAX。所以本專案用 `scripts/jax_compat.py` 把這個函式補回去。
- 所有版本都鎖在 `requirements-duck.txt`，`setup.sh` 已經裝好。

### 步驟

```bash
source .venv-duck/bin/activate

# 1. 確認 JAX 有抓到顯卡（要看到 CudaDevice，不是 CpuDevice）
python -c "import jax; print(jax.devices())"

# 2. 快速測試：約 5 分鐘（大部分時間在第一次 JIT 編譯）
python scripts/duck_train.py --num_timesteps 2000000

# 3. 正式訓練：預設 1.5 億步
python scripts/duck_train.py

# 4. 另開一個終端機看訓練曲線（瀏覽器開 http://localhost:6006）
source .venv-duck/bin/activate
tensorboard --logdir third_party/Open_Duck_Playground/checkpoints

# 5. 用自己訓練的策略走走看（換掉預訓練的 ONNX）
python scripts/duck_walk.py --onnx third_party/Open_Duck_Playground/checkpoints/<時間>_<步數>.onnx
```

說明：

- `scripts/duck_train.py` 只是包裝原專案的 `playground/open_duck_mini_v2/runner.py`。它先載入 `jax_compat.py`，再切到 `third_party/Open_Duck_Playground` 目錄執行，所以參數和原專案一樣（`--help` 可看全部）。
- 每次評估都會存一份 checkpoint 和同名的 `.onnx`，預設放在 `third_party/Open_Duck_Playground/checkpoints/`。要存到別處，`--output_dir` 請給絕對路徑。
- 原專案還會在 `third_party/Open_Duck_Playground/` 產生 `ONNX.onnx` 和 `.tmp/jax_cache`（JIT 編譯快取，第二次啟動會比較快）。這些都已列在原專案的 `.gitignore`，可以忽略。

實測（RTX 5090 筆電、`jax 0.12.1`、`brax 0.14.2`、`mujoco / mujoco-mjx 4.0.1`）：

```
JIT 編譯約 3.5 分鐘，之後約 2 萬步/秒
快速測試 2M 步：跑完、存檔、匯出 ONNX 都正常（這時還不會走路，duck_walk 只走 0.01 m）
正式 150M 步：依上面速度推估約 2 小時
```

### 常見錯誤

| 錯誤訊息 | 原因 | 解法 |
|---|---|---|
| `CUDA-enabled jaxlib is not installed. Falling back to cpu.` | 裝到 CPU 版 JAX，或裝到別的環境 | `uv pip install -p .venv-duck/bin/python -r requirements-duck.txt` |
| `cannot import name 'mjx' from 'mujoco'` | 用了 `uv run`：它會忽略 `.venv-duck`，照原專案設定另建 `.venv`，裝到 Playground 1.0 | **不要用 `uv run`**。刪掉 `third_party/Open_Duck_Playground/.venv`，改用 `python scripts/duck_train.py` |
| `jax.device_put_replicated is deprecated` | 直接執行原專案的 `runner.py`，沒載入相容層 | 改用 `python scripts/duck_train.py` |
| `No module named 'tensorflow'` | 環境是照舊版 README 裝的，缺訓練套件 | 同第一列，重裝 `requirements-duck.txt` |

> 啟動時出現 `Failed to import warp` 是正常的。那是 MuJoCo 4 的選用 GPU 後端，這裡用的是 MJX（JAX），不需要它。

---

## 5b. 走路 + 方向鍵控制（兩種方法對照）

兩台機器人示範兩種完全不同的走路方式，正好是課堂上最好的對照：

| | Open Duck Mini v2 | Mini Pupper 2 |
|---|---|---|
| 方法 | **RL 學出來的策略**（官方預訓練 ONNX） | **傳統步態 + 逆運動學**（不用 RL） |
| 腳本 | `scripts/duck_walk.py` | `scripts/pupper_trot.py` |
| 環境 | `.venv-duck` | `.venv` |
| 控制頻率 | 50 Hz（每 10 個物理步） | 100 Hz（每 5 個物理步） |

```bash
# Mini Pupper 2：小跑步（trot）
source .venv/bin/activate
python scripts/pupper_trot.py

# Open Duck：RL 策略
source .venv-duck/bin/activate
python scripts/duck_walk.py
```

**按鍵**（兩台相同）：

| 按鍵 | 動作 |
|---|---|
| ↑ / ↓ | 加速前進 / 減速、後退（每按一次加一級，共 4 級） |
| ← / → | 左轉 / 右轉 |
| Enter | 停止 |

> 模擬要保持在執行狀態，不要按空白鍵暫停。暫停時方向鍵會被 MuJoCo 檢視器拿去做時間步進。
> 刻意不用字母鍵，因為檢視器已經把字母當快捷鍵（W 線框、C 接觸點…）。

**無頭錄影**（固定路線：直走 4 秒 → 左彎 3 秒 → 原地右轉 2 秒 → 停）：

```bash
MUJOCO_GL=egl python scripts/pupper_trot.py --headless --video pupper_trot.mp4
MUJOCO_GL=egl python scripts/duck_walk.py  --headless --video duck_walk.mp4
```

實測結果（MuJoCo 4.0.1）：

```
Mini Pupper 2：distance 0.65 m, final yaw +52°, stayed up ✅
Open Duck    ：distance 0.43 m, final yaw +31°, stayed up ✅
```

### Mini Pupper 的步態控制器（`pupper_trot.py`）分三層

1. **步態排程**：對角兩腳一組（左前+右後、右前+左後）輪流，每隻腳半個週期著地（stance）、半個週期在空中（swing）。週期 0.3 秒。
2. **腳尖軌跡**：著地時腳尖在身體下方往後滑；擺動時抬腳 2 cm 往前。步幅由指令決定：腳的相對速度 = −(v + ω × r)，所以轉彎時左右腳步幅自動不同。
3. **逆運動學**：腳尖目標（身體座標）→ 三個關節角。用 `mj_jacSite` 取 Jacobian 做阻尼最小平方法數值解，不用推導幾何公式，換機器人也能用。

**課堂討論**：指令 0.2 m/s，實測只有約 0.16 m/s，還會往右偏 8 cm。
- 原因：這是開迴路控制，腳有打滑，沒有用 IMU 回授修正。
- 這正是 Duck 的 RL 策略會吃 IMU 和接觸資訊的原因。
- 學生練習：調 `GAIT` 的週期和抬腳高度，或改伺服 kp，看速度和偏移怎麼變。

### Open Duck 的 RL 策略（`duck_walk.py`）

每 20 ms 做一次：

```
觀測（陀螺儀、加速度計、關節角度與速度、前 3 次動作、腳底接觸、步態相位、速度指令）
  → ONNX 神經網路 → 關節偏移量 × 0.25 + 預設站姿
  → 依伺服最大轉速 5.24 rad/s 限幅 → 位置目標
```

這個策略訓練時的指令範圍是前進 ±0.15 m/s、轉向 ±1.0 rad/s，所以按鍵也限制在這個範圍。超出訓練範圍的指令，策略不保證能走。

---

## 6. 建議的課程順序

| 堂 | 主題 | 腳本 / 指令 |
|---|---|---|
| 0 | 套件拆分、安裝檢查 | `check_install.py` |
| 1 | 看模型：viewer / Studio / 網頁 | `mujoco.viewer`、`mujoco-studio` |
| 2 | PD 站立、推力測試 | `stand_test.py` |
| 3 | URDF → MJCF、伺服模型 | `pupper_urdf_to_mjcf.py` |
| 4 | 走路對照：傳統步態 vs RL 策略、方向鍵操控 | `pupper_trot.py`、`duck_walk.py` |
| 4b | 自己訓練 Duck 的策略（GPU） | `duck_train.py`、`duck_walk.py --onnx` |
| 5 | Sim2sim → Isaac Lab，ONNX → 實機 | 下一單元 |
