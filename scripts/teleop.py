"""Arrow-key teleop shared by both robots (works inside mujoco.viewer.launch_passive).

  ↑ / ↓   : forward / backward speed  (each press = one step, hold to ramp)
  ← / →   : turn left / right
  Enter   : stop (all commands to zero)

The viewer passes GLFW key codes to key_callback. Letters/numbers are avoided
on purpose: MuJoCo's viewer already uses them as shortcuts (W wireframe, C contacts, …).
"""

import numpy as np

KEY_RIGHT, KEY_LEFT, KEY_DOWN, KEY_UP, KEY_ENTER = 262, 263, 264, 265, 257


class Teleop:

  def __init__(self, vx_max: float, wz_max: float, steps: int = 4):
    self.vx_max, self.wz_max = vx_max, wz_max
    self.dvx, self.dwz = vx_max / steps, wz_max / steps
    self.vx = 0.0  # target forward speed [m/s]
    self.wz = 0.0  # target yaw rate [rad/s]

  def key_callback(self, keycode: int):
    if keycode == KEY_UP:
      self.vx += self.dvx
    elif keycode == KEY_DOWN:
      self.vx -= self.dvx
    elif keycode == KEY_LEFT:
      self.wz += self.dwz
    elif keycode == KEY_RIGHT:
      self.wz -= self.dwz
    elif keycode == KEY_ENTER:
      self.vx = self.wz = 0.0
    else:
      return
    self.vx = float(np.clip(self.vx, -self.vx_max, self.vx_max))
    self.wz = float(np.clip(self.wz, -self.wz_max, self.wz_max))
    print(f"\rcommand: vx={self.vx:+.3f} m/s  wz={self.wz:+.2f} rad/s   ", end="", flush=True)


def scripted_command(t: float, vx_max: float, wz_max: float):
  """Fixed demo schedule for headless runs / video recording."""
  if t < 1.0:
    return 0.0, 0.0  # settle
  if t < 5.0:
    return vx_max, 0.0  # straight
  if t < 8.0:
    return 0.6 * vx_max, 0.8 * wz_max  # arc left
  if t < 10.0:
    return 0.0, -wz_max  # spin right in place
  return 0.0, 0.0  # stop
