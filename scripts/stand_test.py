"""站立測試：載入場景 → 套用 home keyframe → 位置控制保持站姿 → 檢查有沒有倒。

兩台機器人共用：
  python scripts/stand_test.py --model <scene.xml>                 # 純模擬，印出結果
  python scripts/stand_test.py --model <scene.xml> --viewer        # 開互動視窗（被動模式）
  MUJOCO_GL=egl python scripts/stand_test.py --model <scene.xml> --video out.mp4   # 無頭錄影
  python scripts/stand_test.py --model <scene.xml> --push 0.5      # 第 1 秒從側面推一下
"""

import argparse
import time

import mujoco
import numpy as np


def actuated_qpos(m: mujoco.MjModel, d: mujoco.MjData) -> np.ndarray:
  """取出每個致動器所驅動關節的目前角度。"""
  return np.array([d.qpos[m.jnt_qposadr[m.actuator_trnid[i, 0]]] for i in range(m.nu)])


def tilt_deg(quat: np.ndarray) -> float:
  """機身 z 軸和世界 z 軸的夾角（度）。"""
  w, x, y, z = quat
  zz = 1 - 2 * (x * x + y * y)
  return float(np.degrees(np.arccos(np.clip(zz, -1, 1))))


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--model", required=True)
  ap.add_argument("--key", default="home")
  ap.add_argument("--seconds", type=float, default=4.0)
  ap.add_argument("--viewer", action="store_true")
  ap.add_argument("--video", default="")
  ap.add_argument("--push", type=float, default=0.0, help="第 1 秒施加的側向推力 [N]")
  args = ap.parse_args()

  m = mujoco.MjModel.from_xml_path(args.model)
  d = mujoco.MjData(m)
  kid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, args.key)
  mujoco.mj_resetDataKeyframe(m, d, kid)
  if not np.any(d.ctrl):  # keyframe 沒寫 ctrl 時，目標 = 目前站姿
    d.ctrl[:] = actuated_qpos(m, d)
  target = d.ctrl.copy()
  mujoco.mj_forward(m, d)
  base = 1  # free joint 所在的機身 body（worldbody 之後第一個）
  z0 = d.xpos[base, 2]

  frames, fps = [], 30
  renderer = mujoco.Renderer(m, 480, 640) if args.video else None
  cam = mujoco.MjvCamera()
  cam.type, cam.trackbodyid, cam.distance, cam.elevation, cam.azimuth = (
      mujoco.mjtCamera.mjCAMERA_TRACKING, base, 0.0, -15, 120)
  span = np.ptp(d.xpos[1:], axis=0).max()  # 機器人本體大小（不含地板）
  cam.distance = max(0.35, 3.0 * span)

  viewer = None
  if args.viewer:
    from mujoco import viewer as mj_viewer  # 需要 mujoco 或 mujoco-core[rendering]
    viewer = mj_viewer.launch_passive(m, d)

  max_tilt, n = 0.0, int(args.seconds / m.opt.timestep)
  for i in range(n):
    t0 = time.time()
    d.ctrl[:] = target
    d.xfrc_applied[base, 1] = args.push if 1.0 <= d.time < 1.1 else 0.0
    mujoco.mj_step(m, d)
    max_tilt = max(max_tilt, tilt_deg(d.xquat[base]))
    if renderer and len(frames) < d.time * fps:
      renderer.update_scene(d, cam)
      frames.append(renderer.render())
    if viewer:
      if not viewer.is_running():
        break
      viewer.sync()
      time.sleep(max(0.0, m.opt.timestep - (time.time() - t0)))

  z = d.xpos[base, 2]
  err = np.abs(actuated_qpos(m, d) - target).max()
  ok = max_tilt < 30 and z > 0.5 * z0
  print(f"模型 {args.model}")
  print(f"  機身高度 {z0:.3f} → {z:.3f} m，最大傾角 {max_tilt:.1f}°，關節最大追蹤誤差 {err:.3f} rad")
  print("  結果：", "站穩 ✅" if ok else "倒了 ❌（檢查 kp / forcerange / 站姿）")

  if renderer:
    import imageio.v3 as iio
    iio.imwrite(args.video, np.stack(frames), fps=fps, codec="libx264")
    print(f"  影片：{args.video}（{len(frames)} 格）")
  if viewer:
    viewer.close()


if __name__ == "__main__":
  main()
