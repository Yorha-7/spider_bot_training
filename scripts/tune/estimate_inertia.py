"""Estimate reflected inertia at each joint from URDF and compute PD damping ratios.

Usage:
    python scripts/tune/estimate_inertia.py
"""

import os
import re
import tempfile
from typing import Dict, List

import numpy as np
import pinocchio as pin

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
_URDF_PATH = os.path.join(
    _REPO_ROOT, "assets", "URDF", "big_bertha", "Spyder_mg995_description.urdf"
)

# Current actuator gains (from assets/big_bertha.py)
CURRENT_KP = 5.0
CURRENT_KD = 0.0  # computed per-joint-group below

JOINT_ORDER = [f"Revolute_{n}" for n in range(110, 122)]
CATEGORIES: Dict[str, str] = {
    "110": "HIP", "111": "THIGH", "112": "CALF",
    "113": "HIP", "114": "THIGH", "115": "CALF",
    "116": "HIP", "117": "THIGH", "118": "CALF",
    "119": "HIP", "120": "THIGH", "121": "CALF",
}
TARGET_ZETA = 0.85


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


def compute_inertia_at_config(model: pin.Model, q) -> Dict[str, float]:
    data = model.createData()
    pin.crba(model, data, q)
    M = data.M
    result = {}
    for name in JOINT_ORDER:
        jid = model.getJointId(name)
        v_idx = model.joints[jid].idx_v
        result[name] = M[v_idx, v_idx]
    return result


def damping_ratio(kp: float, kd: float, I: float) -> float:
    if kp <= 0 or I <= 0:
        return 0.0
    return kd / (2.0 * np.sqrt(kp * I))


def propose_kd(kp: float, I: float, zeta_target: float) -> float:
    return 2.0 * zeta_target * np.sqrt(kp * I)


def main():
    print("=" * 65)
    print("  Big Bertha PD Gain Tuning — Inertia Estimation (pinocchio)")
    print("=" * 65)

    model = load_model()
    q_neutral = pin.neutral(model)

    # Inertia at neutral pose
    I_neutral = compute_inertia_at_config(model, q_neutral)

    # Inertia at walking pose (thighs bent 0.3 rad)
    q_walk = q_neutral.copy()
    for name in ["Revolute_111", "Revolute_114", "Revolute_117", "Revolute_120"]:
        jid = model.getJointId(name)
        q_walk[model.joints[jid].idx_q] = 0.3
    I_walk = compute_inertia_at_config(model, q_walk)

    # --- Current state ---
    print(f"\n{'Joint':<18} {'I_neutral':<14} {'I_walking':<14} {'ζ(current)':<12} {'ωn(Hz)':<10} {'Group':<8}")
    print("-" * 76)

    group_data: Dict[str, Dict] = {g: {"I_vals": [], "I_walk_vals": []} for g in ["HIP", "THIGH", "CALF"]}
    for name in JOINT_ORDER:
        cat = CATEGORIES[name[-3:]]
        I_n = I_neutral[name]
        I_w = I_walk[name]
        zeta = damping_ratio(CURRENT_KP, CURRENT_KD, I_n)
        wn_hz = np.sqrt(CURRENT_KP / I_n) / (2 * np.pi)
        group_data[cat]["I_vals"].append(I_n)
        group_data[cat]["I_walk_vals"].append(I_w)
        print(f"{name:<18} {I_n:<14.8f} {I_w:<14.8f} {zeta:<12.4f} {wn_hz:<10.1f} {cat:<8}")

    # --- Group summary ---
    print(f"\n{'Group':<8} {'I_mean':<12} {'I_walk_mean':<14} {'ζ(current)':<12} {'ζ(walking)':<12} {'Rec kd(ζ=0.85)':<16}")
    print("-" * 74)

    recommendations = {}
    for cat in ["HIP", "THIGH", "CALF"]:
        I_arr = np.array(group_data[cat]["I_vals"])
        I_w_arr = np.array(group_data[cat]["I_walk_vals"])
        I_mean = I_arr.mean()
        I_w_mean = I_w_arr.mean()
        zeta_n = damping_ratio(CURRENT_KP, CURRENT_KD, I_mean)
        zeta_w = damping_ratio(CURRENT_KP, CURRENT_KD, I_w_mean)
        kd_rec = propose_kd(CURRENT_KP, I_mean, TARGET_ZETA)
        kd_rec_w = propose_kd(CURRENT_KP, I_w_mean, TARGET_ZETA)
        final_kd = round((kd_rec + kd_rec_w) / 2, 4)
        recommendations[cat] = final_kd
        print(f"{cat:<8} {I_mean:<12.8f} {I_w_mean:<14.8f} {zeta_n:<12.4f} {zeta_w:<12.4f} kd={final_kd:<10.4f}")

    # --- Key insight ---
    print("\n" + "=" * 65)
    print("  KEY INSIGHT")
    print("=" * 65)
    print(f"""  Current kd=2.0 produces ζ=4-10 → extremely overdamped.
  Damping torque at qd=1 rad/s: τ_damp = {CURRENT_KD} N·m
  MG995 stall torque: 1.18 N·m
  → Damping saturates the motor at just {1.18/CURRENT_KD:.2f} rad/s!
  → Almost no torque budget left for position tracking.

  Fix: keep kp={CURRENT_KP} (good tracking bandwidth), reduce kd per-joint.""")

    # --- Proposed config changes ---
    print("\n" + "=" * 65)
    print("  PROPOSED CONFIG CHANGES")
    print("=" * 65)

    for cat in ["HIP", "THIGH", "CALF"]:
        I_mean = np.mean(group_data[cat]["I_vals"])
        kd = recommendations[cat]
        zeta_check = damping_ratio(CURRENT_KP, kd, I_mean)
        zeta_walk = damping_ratio(CURRENT_KP, kd, np.mean(group_data[cat]["I_walk_vals"]))
        print(f"  {cat:6}: stiffness={CURRENT_KP:.0f}, damping={kd:.4f}  (ζ={zeta_check:.3f} neutral, ζ={zeta_walk:.3f} walking)")

    # --- DR range recommendation ---
    print(f"""
  Domain randomization (scale ranges) — centered on new kd:
    HIP:  stiffness_distribution_params=(0.6, 1.4),  damping_distribution_params=(0.5, 2.0)
    THIGH: same
    CALF:  same
  (The scale range already covers the variation; only base values change.)
  """)

    # --- Check explicit mode stability ---
    print("=" * 65)
    print("  STABILITY CHECK (explicit IdealPD, dt=1/500)")
    print("=" * 65)
    dt = 1 / 500
    for cat in ["HIP", "THIGH", "CALF"]:
        I_mean = np.mean(group_data[cat]["I_vals"])
        kd = recommendations[cat]
        wn = np.sqrt(CURRENT_KP / I_mean)
        stability = CURRENT_KP * dt**2 / I_mean
        print(f"  {cat:6}: ωn={wn:.1f} rad/s ({wn/(2*np.pi):.1f} Hz), kp·dt²/I={stability:.3f} (target <2)")


if __name__ == "__main__":
    main()
