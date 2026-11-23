"""Open Duck Mini v2: walk with the official pretrained RL policy (ONNX), arrow-key control.

Needs the .venv-duck environment (playground==0.0.5 + onnxruntime) and two repos:
  third_party/Open_Duck_Playground   (sim model + inference helpers)
  third_party/Open_Duck_Mini         (BEST_WALK_ONNX_2.onnx, the pretrained policy)

  python scripts/duck_walk.py                         # viewer + arrow keys
  MUJOCO_GL=egl python scripts/duck_walk.py --headless --video duck_walk.mp4   # scripted demo

How it works (one control step = 10 physics steps = 50 Hz):
  observation (IMU, joint angles/velocities, last actions, foot contacts, gait phase, command)
  → ONNX policy → joint offsets × 0.25 + default pose → clipped by servo speed → position targets
"""

import argparse
import pathlib
import sys
import time

import mujoco
import numpy as np

from teleop import Teleop, scripted_command

ROOT = pathlib.Path(__file__).resolve().parents[1]
DUCK_REPO = ROOT / "third_party" / "Open_Duck_Playground"
sys.path.insert(0, str(DUCK_REPO))
from playground.open_duck_mini_v2.mujoco_infer import MjInfer  # noqa: E402

VX_MAX, WZ_MAX = 0.15, 1.0  # the ranges the policy was trained on


def policy_step(duck: MjInfer):
  """One 50 Hz control update (same logic as MjInfer.run, without the viewer)."""
  duck.imitation_i = (duck.imitation_i + duck.phase_frequency_factor) % duck.PRM.nb_steps_in_period
  ph = duck.imitation_i / duck.PRM.nb_steps_in_period * 2 * np.pi
  duck.imitation_phase = np.array([np.cos(ph), np.sin(ph)])
  action = duck.policy.infer(duck.get_obs(duck.data, duck.commands))
  duck.last_last_last_action = duck.last_last_action.copy()
  duck.last_last_action = duck.last_action.copy()
  duck.last_action = action.copy()
  targets = duck.default_actuator + action * duck.action_scale
  dmax = duck.max_motor_velocity * duck.sim_dt * duck.decimation
  targets = np.clip(targets, duck.prev_motor_targets - dmax, duck.prev_motor_targets + dmax)
  duck.prev_motor_targets = duck.motor_targets = targets.copy()
  duck.data.ctrl[:] = targets


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--onnx", default=str(ROOT / "third_party/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx"))
  ap.add_argument("--scene", default=str(DUCK_REPO / "playground/open_duck_mini_v2/xmls/scene_flat_terrain.xml"))
  ap.add_argument("--headless", action="store_true")
  ap.add_argument("--video", default="")
  ap.add_argument("--seconds", type=float, default=12.0)
  args = ap.parse_args()

  duck = MjInfer(args.scene, str(DUCK_REPO / "playground/open_duck_mini_v2/data/polynomial_coefficients.pkl"),
                 args.onnx, standing=False)
  m, d = duck.model, duck.data
  teleop = Teleop(VX_MAX, WZ_MAX)

  def set_command(vx, wz):
    duck.commands[0], duck.commands[1], duck.commands[2] = vx, 0.0, wz

  if not args.headless:
    from mujoco import viewer as mj_viewer
    print(__doc__.split("How it works")[0])
    print("↑/↓ speed   ←/→ turn   Enter stop")
    with mj_viewer.launch_passive(m, d, key_callback=teleop.key_callback) as v:
      step = 0
      while v.is_running():
        t0 = time.time()
        if step % duck.decimation == 0:
          set_command(teleop.vx, teleop.wz)
          policy_step(duck)
        mujoco.mj_step(m, d)
        step += 1
        v.sync()
        time.sleep(max(0.0, m.opt.timestep - (time.time() - t0)))
    return

  # ---- headless scripted demo ----
  renderer = mujoco.Renderer(m, 480, 640) if args.video else None
  cam = mujoco.MjvCamera()
  cam.type, cam.trackbodyid, cam.distance, cam.elevation, cam.azimuth = (
      mujoco.mjtCamera.mjCAMERA_TRACKING, 1, 0.8, -20, 135)
  frames, start = [], d.qpos[:2].copy()
  base = m.jnt_bodyid[duck._floating_base_id]
  fell = False
  for step in range(int(args.seconds / m.opt.timestep)):
    if step % duck.decimation == 0:
      set_command(*scripted_command(d.time, VX_MAX, WZ_MAX))
      policy_step(duck)
    mujoco.mj_step(m, d)
    up = d.xmat[base].reshape(3, 3)[2, 2]  # z-axis alignment of the trunk
    fell |= up < 0.5
    if renderer and len(frames) < d.time * 30:
      renderer.update_scene(d, cam)
      frames.append(renderer.render())
  w, x, y, z = d.qpos[3:7]
  yaw = np.degrees(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
  print(f"distance {np.linalg.norm(d.qpos[:2] - start):.2f} m, final yaw {yaw:+.0f}°, "
        f"{'FELL ❌' if fell else 'stayed up ✅'}")
  if renderer:
    import imageio.v3 as iio
    iio.imwrite(args.video, np.stack(frames), fps=30, codec="libx264")
    print(f"video: {args.video}")


if __name__ == "__main__":
  main()
