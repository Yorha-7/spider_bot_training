"""Find correct default joint angles and base height for Big Bertha.

The current defaults (thighs=0.5 all around) produce asymmetric stance because
left and right legs have opposite hip yaw orientations. This script searches for
thigh angles that equalize foot heights, then computes base z.
"""

import os
import re
import tempfile

import numpy as np
import pinocchio as pin

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
_URDF_PATH = os.path.join(
    _REPO_ROOT, "assets", "URDF", "big_bertha", "Spyder_mg995_description.urdf"
)

# Default joint positions from source/big_bertha/big_bertha/assets/big_bertha.py
_DEFAULT_HIP = 0.0
_DEFAULT_THIGH = 0.5
_DEFAULT_CALF = 0.0

# Foot links and their approximate foot tip offset (visual origin + mesh extent)
FOOT_VISUAL_OFFSETS = {
    "arm_c_1_1": np.array([0.01, -0.145, -0.053]),   # FR
    "arm_c_2_1": np.array([0.012, -0.1455, -0.040]),  # FL
    "arm_c_3_1": np.array([0.012, -0.1455, -0.040]),  # RL
    "arm_c_4_1": np.array([0.01, -0.145, -0.053]),    # RR
}
MESH_FOOT_OFFSET = np.array([0.0, -0.03, -0.01])   # approximate mesh extents
FOOT_TIP_OFFSETS = {k: v + MESH_FOOT_OFFSET for k, v in FOOT_VISUAL_OFFSETS.items()}

FOOT_NAMES = ["arm_c_1_1", "arm_c_2_1", "arm_c_3_1", "arm_c_4_1"]
# Joint names in order: 110=FR, 113=FL, 116=RR, 119=RL
HIP_NAMES = ["Revolute_110", "Revolute_113", "Revolute_116", "Revolute_119"]
THIGH_NAMES = ["Revolute_111", "Revolute_114", "Revolute_117", "Revolute_120"]
CALF_NAMES = ["Revolute_112", "Revolute_115", "Revolute_118", "Revolute_121"]
# Index into FOOT_NAMES for each leg in the same order as above
# FR=0, FL=1, RR=3, RL=2 (based on env mapping)
LEG_TO_FOOT = [0, 1, 3, 2]


def load_model() -> pin.Model:
    with open(_URDF_PATH) as f:
        content = f.read()
    content = re.sub(r'<xacro:include[^>]*/>', '', content)
    content = content.replace('xmlns:xacro="http://www.ros.org/wiki/xacro"', '')
    with tempfile.NamedTemporaryFile(suffix='.urdf', mode='w', delete=False) as f:
        f.write(content)
        tmp_path = f.name
    try:
        model = pin.Model()
        pin.buildModelFromUrdf(tmp_path, pin.JointModelFreeFlyer(), model)
    finally:
        os.unlink(tmp_path)
    return model


def foot_tip_zs(model, data, q):
    """Return foot tip z for each of the 4 feet."""
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    zs = []
    for fn in FOOT_NAMES:
        fid = model.getFrameId(fn)
        frame_pos = data.oMf[fid].translation
        R = data.oMf[fid].rotation
        tip_world = frame_pos + R @ FOOT_TIP_OFFSETS[fn]
        zs.append(tip_world[2])
    return np.array(zs)


def main():
    model = load_model()
    data = model.createData()

    # --- Current default pose ---
    q = pin.neutral(model)
    for name in HIP_NAMES + THIGH_NAMES + CALF_NAMES:
        jid = model.getJointId(name)
        if name in THIGH_NAMES:
            q[model.joints[jid].idx_q] = _DEFAULT_THIGH
        else:
            q[model.joints[jid].idx_q] = 0.0

    z_current = foot_tip_zs(model, data, q)

    print("=" * 65)
    print("  Big Bertha — Initial Position Fix")
    print("=" * 65)
    print(f"\nCURRENT defaults (ALL thighs={_DEFAULT_THIGH} rad):")
    for i, fn in enumerate(FOOT_NAMES):
        print(f"  {fn}: tip z = {z_current[i]:+.4f} m")
    print(f"  Range: {z_current.min():.4f} to {z_current.max():+.4f} m "
          f"(Δ = {z_current.max() - z_current.min():.4f} m)")

    # --- Find left/right thigh angles that equalize foot heights ---
    # The left legs (FL, RL) point up; right legs (FR, RR) point down.
    # We need DIFFERENT thigh angles for left vs right.
    # Let's do a grid search over left_thigh and right_thigh.

    print("\nSearching for balanced thigh angles...")
    best_std = float('inf')
    best_pair = (0.5, 0.5)
    best_zs = None

    for lt in np.arange(0.0, 1.6, 0.01):   # left thighs (FL=leg1, RL=leg3)
        for rt in np.arange(0.0, 1.6, 0.01):  # right thighs (FR=leg0, RR=leg2)
            q_try = pin.neutral(model)
            jid = model.getJointId("Revolute_114")
            q_try[model.joints[jid].idx_q] = lt  # FL thigh
            jid = model.getJointId("Revolute_120")
            q_try[model.joints[jid].idx_q] = lt  # RL thigh
            jid = model.getJointId("Revolute_111")
            q_try[model.joints[jid].idx_q] = rt  # FR thigh
            jid = model.getJointId("Revolute_117")
            q_try[model.joints[jid].idx_q] = rt  # RR thigh

            zs = foot_tip_zs(model, data, q_try)
            z_std = zs.std()
            if z_std < best_std:
                best_std = z_std
                best_pair = (lt, rt)
                best_zs = zs

    print(f"  Best: left_thigh={best_pair[0]:.2f}, right_thigh={best_pair[1]:.2f}")
    print(f"  Foot z std = {best_std:.4f} m")
    for i, fn in enumerate(FOOT_NAMES):
        print(f"  {fn}: tip z = {best_zs[i]:+.4f} m")
    print(f"  Range: {best_zs.min():.4f} to {best_zs.max():+.4f} m "
          f"(Δ = {best_zs.max() - best_zs.min():.4f} m)")

    # Base height so lowest foot is at z=0
    base_z = -best_zs.min()
    print(f"\n  Recommended init_state pos.z = {base_z:.4f} m")

    # --- Also try with hip offset ---
    print("\n\nTrying with hip joint adjustment + equal thigh angles...")
    best_std2 = float('inf')
    best_hip = 0.0
    best_thigh = 0.5
    best_zs2 = None

    for th in np.arange(0.0, 1.6, 0.01):
        for hp in np.linspace(-0.3, 0.3, 61):  # small hip correction
            q_try = pin.neutral(model)
            # Left legs (FL=leg1, RL=leg3) — mirror the hip
            for hname in ["Revolute_113", "Revolute_119"]:
                jid = model.getJointId(hname)
                q_try[model.joints[jid].idx_q] = hp
            for tname in ["Revolute_114", "Revolute_120"]:
                jid = model.getJointId(tname)
                q_try[model.joints[jid].idx_q] = th
            # Right legs (FR=leg0, RR=leg2)
            for hname in ["Revolute_110", "Revolute_116"]:
                jid = model.getJointId(hname)
                q_try[model.joints[jid].idx_q] = -hp
            for tname in ["Revolute_111", "Revolute_117"]:
                jid = model.getJointId(tname)
                q_try[model.joints[jid].idx_q] = th

            zs = foot_tip_zs(model, data, q_try)
            z_std = zs.std()
            if z_std < best_std2:
                best_std2 = z_std
                best_pair2 = (hp, th)
                best_zs2 = zs

    print(f"  Best: hip={best_pair2[0]:.3f}, thigh={best_pair2[1]:.2f}")
    print(f"  Foot z std = {best_std2:.4f} m")
    for i, fn in enumerate(FOOT_NAMES):
        print(f"  {fn}: tip z = {best_zs2[i]:+.4f} m")
    base_z2 = -best_zs2.min()
    print(f"  Recommended init_state pos.z = {base_z2:.4f} m")

    # --- Final recommended config ---
    print("\n" + "=" * 65)
    print("  RECOMMENDED CHANGE")
    print("=" * 65)

    if best_std <= best_std2:
        lt, rt = best_pair
        print(f"\n  Use DIFFERENT left/right thigh angles:")
        print(f"    Left-side thighs (FL, RL): {lt:.2f} rad")
        print(f"    Right-side thighs (FR, RR): {rt:.2f} rad")
        print(f"    All hips: 0.0 rad")
        print(f"    All calves: 0.0 rad")
        print(f"    pos.z: {base_z:.4f} m")
        print(f"\n  Updated joint_pos dict:")
        print(f'    "Revolute_110": 0.0,   # FR hip')
        print(f'    "Revolute_111": {rt:.2f},   # FR thigh')
        print(f'    "Revolute_112": 0.0,   # FR calf')
        print(f'    "Revolute_113": 0.0,   # FL hip')
        print(f'    "Revolute_114": {lt:.2f},   # FL thigh')
        print(f'    "Revolute_115": 0.0,   # FL calf')
        print(f'    "Revolute_116": 0.0,   # RR hip')
        print(f'    "Revolute_117": {rt:.2f},   # RR thigh')
        print(f'    "Revolute_118": 0.0,   # RR calf')
        print(f'    "Revolute_119": 0.0,   # RL hip')
        print(f'    "Revolute_120": {lt:.2f},   # RL thigh')
        print(f'    "Revolute_121": 0.0,   # RL calf')
    else:
        hp, th = best_pair2
        print(f"\n  Use hip offset + equal thigh angles:")
        print(f"    Left-side hips (FL, RL): {hp:.3f} rad")
        print(f"    Right-side hips (FR, RR): {-hp:.3f} rad")
        print(f"    All thighs: {th:.2f} rad")
        print(f"    All calves: 0.0 rad")
        print(f"    pos.z: {base_z2:.4f} m")

    print("\nDone.")


if __name__ == "__main__":
    main()
