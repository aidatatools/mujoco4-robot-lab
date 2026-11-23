"""Mini Pupper 2: classic trot gait (no RL), arrow-key control.

  python scripts/pupper_trot.py                           # viewer + arrow keys
  MUJOCO_GL=egl python scripts/pupper_trot.py --headless --video pupper_trot.mp4   # scripted demo

The controller has three layers, the same idea as the Stanford Pupper / CHAMP:
  1. Gait scheduler: diagonal pairs (LF+RB, RF+LB) alternate. Each leg spends half of
     the period on the ground (stance), half in the air (swing).
  2. Foot trajectory: in stance the foot slides backward under the body; in swing it
     lifts and moves forward. Step length comes from the command (vx, yaw rate).
  3. Inverse kinematics: foot target (body frame) → 3 joint angles per leg, solved
     numerically with MuJoCo's Jacobian (mj_jacSite), then sent to the position servos.
"""

import argparse
import pathlib
import time

import mujoco
import numpy as np

from teleop import Teleop, scripted_command

ROOT = pathlib.Path(__file__).resolve().parents[1]
LEGS = ["lf", "rf", "lb", "rb"]
PHASE_OFFSET = {"lf": 0.0, "rb": 0.0, "rf": 0.5, "lb": 0.5}  # trot = diagonal pairs

GAIT = dict(
    period=0.30,  # one full gait cycle [s]
    duty=0.5,  # fraction of the cycle on the ground
    step_height=0.020,  # swing foot clearance [m]
    control_dt=0.01,  # IK / target update rate [s] (100 Hz)
)
VX_MAX, WZ_MAX = 0.20, 1.5  # m/s, rad/s


class TrotController:

  def __init__(self, m: mujoco.MjModel):
    self.m = m
    self.kd = mujoco.MjData(m)  # separate "kinematics only" data, body fixed at origin
    key = m.key("home")
    self.kd.qpos[:] = key.qpos
    self.kd.qpos[:7] = [0, 0, 0, 1, 0, 0, 0]
    mujoco.mj_kinematics(m, self.kd)
    self.site = {leg: m.site(f"foot_{leg}").id for leg in LEGS}
    self.nominal = {leg: self.kd.site_xpos[self.site[leg]].copy() for leg in LEGS}  # stance feet
    self.qadr, self.dadr, self.act = {}, {}, {}
    for leg in LEGS:
      joints = [m.joint(n) for n in (f"base_{leg}1", f"{leg}1_{leg}2", f"{leg}2_{leg}3")]
      self.qadr[leg] = np.array([j.qposadr[0] for j in joints])
      self.dadr[leg] = np.array([j.dofadr[0] for j in joints])
      self.act[leg] = np.array([m.actuator(j.name).id for j in joints])
    self.jacp = np.zeros((3, m.nv))
    self.cmd = np.zeros(2)  # smoothed (vx, wz)

  def foot_target(self, leg: str, t: float) -> np.ndarray:
    T, duty, h = GAIT["period"], GAIT["duty"], GAIT["step_height"]
    p0 = self.nominal[leg]
    vx, wz = self.cmd
    # foot velocity relative to the body during stance = -(v + w × r)
    stance_len = np.array([vx - wz * p0[1], wz * p0[0], 0.0]) * duty * T
    moving = np.hypot(vx, wz * 0.1) > 1e-3
    ph = (t / T + PHASE_OFFSET[leg]) % 1.0
    if ph < duty:  # stance: slide from front to back
      s = ph / duty
      return p0 + stance_len * (0.5 - s)
    s = (ph - duty) / (1 - duty)  # swing: back to front, lifted
    s_smooth = 0.5 - 0.5 * np.cos(np.pi * s)
    lift = h * np.sin(np.pi * s) if moving else 0.0
    return p0 + stance_len * (s_smooth - 0.5) + np.array([0, 0, lift])

  def solve_ik(self, leg: str, target: np.ndarray, iters: int = 3, damping: float = 1e-4):
    qa, da = self.qadr[leg], self.dadr[leg]
    for _ in range(iters):
      mujoco.mj_kinematics(self.m, self.kd)
      mujoco.mj_comPos(self.m, self.kd)
      err = target - self.kd.site_xpos[self.site[leg]]
      mujoco.mj_jacSite(self.m, self.kd, self.jacp, None, self.site[leg])
      J = self.jacp[:, da]
      self.kd.qpos[qa] += np.linalg.solve(J.T @ J + damping * np.eye(3), J.T @ err)
    return self.kd.qpos[qa]

  def update(self, d: mujoco.MjData, vx: float, wz: float, t: float):
    # ramp the command so key presses don't jolt the robot
    a = 0.05
    self.cmd = (1 - a) * self.cmd + a * np.array([vx, wz])
    for leg in LEGS:
      d.ctrl[self.act[leg]] = self.solve_ik(leg, self.foot_target(leg, t))


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--model", default=str(ROOT / "models/mini_pupper_2/scene.xml"))
  ap.add_argument("--headless", action="store_true")
  ap.add_argument("--video", default="")
  ap.add_argument("--seconds", type=float, default=12.0)
  args = ap.parse_args()

  m = mujoco.MjModel.from_xml_path(args.model)
  d = mujoco.MjData(m)
  mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
  ctrl = TrotController(m)
  teleop = Teleop(VX_MAX, WZ_MAX)
  every = round(GAIT["control_dt"] / m.opt.timestep)

  if not args.headless:
    from mujoco import viewer as mj_viewer
    print(__doc__.split("The controller")[0])
    print("↑/↓ speed   ←/→ turn   Enter stop")
    with mj_viewer.launch_passive(m, d, key_callback=teleop.key_callback) as v:
      step = 0
      while v.is_running():
        t0 = time.time()
        if step % every == 0:
          ctrl.update(d, teleop.vx, teleop.wz, d.time)
        mujoco.mj_step(m, d)
        step += 1
        v.sync()
        time.sleep(max(0.0, m.opt.timestep - (time.time() - t0)))
    return

  # ---- headless scripted demo ----
  renderer = mujoco.Renderer(m, 480, 640) if args.video else None
  cam = mujoco.MjvCamera()
  cam.type, cam.trackbodyid, cam.distance, cam.elevation, cam.azimuth = (
      mujoco.mjtCamera.mjCAMERA_TRACKING, 1, 0.55, -20, 135)
  frames, start, fell = [], d.qpos[:2].copy(), False
  for step in range(int(args.seconds / m.opt.timestep)):
    if step % every == 0:
      ctrl.update(d, *scripted_command(d.time, VX_MAX, WZ_MAX), d.time)
    mujoco.mj_step(m, d)
    fell |= d.xmat[1].reshape(3, 3)[2, 2] < 0.5
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
