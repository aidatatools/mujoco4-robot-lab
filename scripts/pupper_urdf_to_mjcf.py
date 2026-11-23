"""Mini Pupper 2：URDF(xacro 展開後) → MuJoCo 4 MJCF 場景。

用法：
  python scripts/pupper_urdf_to_mjcf.py \
      --urdf models/mini_pupper_2/mini_pupper_2.urdf \
      --meshdir ~/mini_pupper_ros/mini_pupper_description/meshes/mini_pupper_2 \
      --out models/mini_pupper_2/scene.xml

注意：
  * URDF 要用 `use_gazebo_hardware:=false` 展開，否則質量 ×30、慣量 ×100（那是給 Gazebo ODE 用的權宜值）。
  * 下方 SERVO 參數是「教學起始值」，不是實測值。要做 sim2real 必須用實機做系統辨識後再改。
"""

import argparse
import pathlib
import re
import shutil

import mujoco

# ---- 伺服馬達模型（教學起始值，請依實機校正）----
SERVO = dict(
    kp=4.0,  # 位置控制增益 [N·m/rad]
    kv=0.08,  # 速度阻尼（寫在 actuator 上）[N·m·s/rad]
    forcerange=0.35,  # 最大扭力 [N·m]，小型伺服約 0.2–0.4
    damping=0.01,  # 關節被動阻尼 [N·m·s/rad]（URDF 的 5.0 是佔位值，太大）
    armature=0.0005,  # 反射轉子慣量 [kg·m²]，小連桿一定要給，否則數值不穩
    frictionloss=0.005,  # 靜摩擦 [N·m]
)

# 站姿（取自 mini_pupper_ros ros2_control 的 initial_value）
HOME = {"abduction": 0.0, "hip": 0.994, "knee": -1.767}
LEGS = ["lf", "rf", "lb", "rb"]


def joint_role(name: str) -> str:
  if name.startswith("base_"):
    return "abduction"  # base_lf1：髖外展
  if re.match(r"^[lr][fb]1_", name):
    return "hip"  # lf1_lf2：大腿
  return "knee"  # lf2_lf3：小腿


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--urdf", required=True)
  ap.add_argument("--meshdir", required=True)
  ap.add_argument("--out", required=True)
  ap.add_argument("--z0", type=float, default=0.12, help="初始機身高度 [m]")
  args = ap.parse_args()

  out = pathlib.Path(args.out).resolve()
  (out.parent / "meshes").mkdir(parents=True, exist_ok=True)
  meshdir = pathlib.Path(args.meshdir).expanduser()

  urdf = pathlib.Path(args.urdf).read_text()
  # package://.../xxx.stl → xxx.stl，網格統一複製到 out/meshes/
  for f in set(re.findall(r'<mesh\s+filename="([^"]+)"', urdf)):
    base = pathlib.Path(f).name
    shutil.copy(meshdir / base, out.parent / "meshes" / base)
    urdf = urdf.replace(f, base)
  # MuJoCo 的 URDF 擴充：保留視覺幾何、修正過小慣量
  urdf = urdf.replace(
      "<robot name=\"mini_pupper_2\">",
      "<robot name=\"mini_pupper_2\"><mujoco><compiler meshdir=\"meshes\""
      " balanceinertia=\"true\" discardvisual=\"false\""
      " fusestatic=\"true\"/></mujoco>",
      1,
  )
  tmp = out.parent / "_tmp_robot.urdf"
  tmp.write_text(urdf)

  spec = mujoco.MjSpec.from_file(str(tmp))
  tmp.unlink()
  spec.modelname = "mini_pupper_2"
  spec.option.timestep = 0.002
  spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST

  # 1) 機身加 free joint（URDF 根連桿預設是焊在世界上）
  base = spec.worldbody.first_body()
  base.pos = [0, 0, args.z0]
  base.add_freejoint(name="floating_base")

  # 2) 關節參數 + 位置控制器
  joints = [j for j in spec.joints if j.type == mujoco.mjtJoint.mjJNT_HINGE]
  home_q = []
  for j in joints:
    # MuJoCo 4：damping/stiffness 變成多項式係數 [c0, c1, c2]，只填一次項
    j.damping = [SERVO["damping"], 0.0, 0.0]
    j.armature = SERVO["armature"]
    j.frictionloss = SERVO["frictionloss"]
    j.range = [-3.14, 3.14]
    a = spec.add_actuator(name=j.name, target=j.name)
    a.trntype = mujoco.mjtTrn.mjTRN_JOINT
    a.set_to_position(kp=SERVO["kp"], kv=SERVO["kv"])
    a.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
    a.forcerange = [-SERVO["forcerange"], SERVO["forcerange"]]
    home_q.append(HOME[joint_role(j.name)])

  # 2b) 腳尖 site（步態 IK 用）：小腿座標系下 (-0.0059, 0, -0.062)，由站立時的接觸點量得
  for leg in LEGS:
    spec.body(f"{leg}3").add_site(name=f"foot_{leg}", pos=[-0.0059, 0, -0.062], size=[0.006, 0, 0],
                                  rgba=[1, 0.3, 0.3, 1])

  # 3) 地板、光源、相機
  spec.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D,
                   builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                   rgb1=[.25, .3, .35], rgb2=[.35, .4, .45], width=300, height=300)
  spec.add_material(name="grid", textures=["", "grid"], texrepeat=[20, 20])
  spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
                          size=[5, 5, .05], material="grid", friction=[1.0, .005, .0001])
  spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                   builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                   rgb1=[.6, .8, 1], rgb2=[1, 1, 1], width=512, height=3072)
  spec.worldbody.add_light(pos=[0, 0, 2], dir=[0, 0, -1], castshadow=False,
                           type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
  spec.worldbody.add_light(pos=[0, 0, 1.0], dir=[0, 0, -1], mode=mujoco.mjtCamLight.mjCAMLIGHT_TRACKCOM,
                           type=mujoco.mjtLightType.mjLIGHT_SPOT, cutoff=60)
  spec.worldbody.add_camera(name="side", pos=[0.0, -0.6, 0.25], xyaxes=[1, 0, 0, 0, 0.4, 1])

  # 4) 站姿 keyframe（qpos = 機身 7 維 + 12 關節；ctrl = 12 關節目標）
  key = spec.add_key(name="home")
  key.qpos = [0, 0, args.z0, 1, 0, 0, 0] + home_q
  key.ctrl = home_q

  m = spec.compile()  # 先編譯一次確認沒錯
  out.write_text(spec.to_xml())
  print(f"OK → {out}  nq={m.nq} nv={m.nv} nu={m.nu} mass={sum(m.body_mass):.3f} kg")
  for j, q in zip(joints, home_q):
    print(f"  {j.name:10s} {joint_role(j.name):9s} home={q:+.3f}")


if __name__ == "__main__":
  main()
