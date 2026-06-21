"""Step-response test for big_bertha PD actuators.

Applies position step commands to individual joints in Isaac Lab simulation
and logs position, velocity, and effort for empirical PD response analysis.

Usage:
    python scripts/tune/actuator_step_response.py [--headless] [--joint Revolute_110]
"""

import argparse
import csv
import os
import time
from datetime import datetime

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationCfg

from big_bertha.assets.big_bertha import BIG_BERTHA_CFG

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

# Joints to test (one from each group)
TEST_JOINTS = {
    "hip": "Revolute_110",
    "thigh": "Revolute_111",
    "calf": "Revolute_112",
}
ALL_JOINTS = [f"Revolute_{n}" for n in range(110, 122)]
STEP_AMPLITUDE = 0.5  # rad
SETTLE_TIME = 0.5  # seconds to settle at start
RECORD_TIME = 1.0  # seconds to record after step


def run_step_test(joint_name: str, headless: bool = True) -> str:
    """Run step response test on a single joint and save CSV data."""
    joint_idx = ALL_JOINTS.index(joint_name)
    group_name = [k for k, v in TEST_JOINTS.items() if v == joint_name][0]

    # Simulation config — use explicit (IdealPD) mode dt for accuracy
    sim_cfg = SimulationCfg(
        dt=1 / 500,
        render_interval=1,
        device="cuda:0" if not headless else "cpu",
    )

    # Initialize simulation
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_sim_params()

    # Spawn the robot
    robot_cfg = BIG_BERTHA_CFG.replace(prim_path="/World/envs/env_0/Robot")
    robot = Articulation(robot_cfg)
    robot.spawn()

    # Setup scene
    sim.reset()
    robot.reset()

    # Get joint indices
    dof_names = robot.data.dof_names
    local_idx = dof_names.index(joint_name)
    n_dof = len(dof_names)

    # Data collection
    data = {
        "time": [],
        "position": [],
        "velocity": [],
        "effort": [],
        "position_command": [],
    }

    # Phase 1: Settle at zero
    t_settle = int(SETTLE_TIME / sim_cfg.dt)
    for _ in range(t_settle):
        joint_pos = torch.zeros(1, n_dof, device=sim.device)
        joint_vel = torch.zeros(1, n_dof, device=sim.device)
        robot.write_joint_state_to_sim(joint_pos, joint_vel, None)
        robot.set_joint_position_target(joint_pos)
        sim.step()
        robot.update(sim.cfg.dt)

    # Phase 2: Record settling baseline
    t_record = int(RECORD_TIME / sim_cfg.dt)
    for step_i in range(t_record):
        # Apply step at t=0
        joint_pos_target = torch.zeros(1, n_dof, device=sim.device)
        joint_pos_target[0, local_idx] = STEP_AMPLITUDE
        robot.set_joint_position_target(joint_pos_target)

        sim.step()
        robot.update(sim.cfg.dt)

        # Record
        data["time"].append(step_i * sim_cfg.dt)
        data["position"].append(robot.data.joint_pos[0, local_idx].item())
        data["velocity"].append(robot.data.joint_vel[0, local_idx].item())
        data["effort"].append(robot.data.applied_torque[0, local_idx].item())
        data["position_command"].append(joint_pos_target[0, local_idx].item())

    sim.stop()

    # Save CSV
    out_dir = os.path.join(_REPO_ROOT, "scripts", "tune", "results")
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(out_dir, f"step_response_{group_name}_{joint_name}_{timestamp}.csv")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=data.keys())
        writer.writeheader()
        for i in range(len(data["time"])):
            writer.writerow({k: v[i] for k, v in data.items()})

    # Compute metrics
    pos = np.array(data["position"])
    cmd = np.array(data["position_command"])
    t = np.array(data["time"])
    steady = pos[-int(0.2 / sim_cfg.dt):]  # last 200 ms
    steady_val = np.mean(steady)

    # Rise time: 10% to 90%
    final_val = steady_val
    min_val = pos[0]
    range_10 = min_val + 0.1 * (final_val - min_val)
    range_90 = min_val + 0.9 * (final_val - min_val)
    t_10 = t[np.where(pos >= range_10)][0] if np.any(pos >= range_10) else t[-1]
    t_90 = t[np.where(pos >= range_90)][0] if np.any(pos >= range_90) else t[-1]
    rise_time = t_90 - t_10

    # Overshoot
    overshoot_pct = max(0, (np.max(pos) - final_val) / (final_val - min_val) * 100)

    # Settling time (within 2%)
    band = 0.02 * (final_val - min_val)
    settled = np.where(np.abs(pos - final_val) > band)[0]
    settle_time = t[settled[-1]] - t[0] if len(settled) > 0 and settled[-1] > 0 else 0.0

    # Steady-state error
    ss_error = abs(steady_val - cmd[-1])

    metrics = {
        "joint": joint_name,
        "group": group_name,
        "rise_time_s": round(rise_time, 4),
        "overshoot_pct": round(overshoot_pct, 2),
        "settle_time_s": round(settle_time, 4),
        "steady_state_value": round(steady_val, 4),
        "steady_state_error": round(ss_error, 4),
        "csv_path": csv_path,
    }

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Big Bertha actuator step-response test")
    parser.add_argument("--headless", action="store_true", default=True,
                        help="Run headless (default: True)")
    parser.add_argument("--joint", type=str, default=None,
                        help="Specific joint to test (default: test all 3 groups)")
    args = parser.parse_args()

    if args.joint:
        test_joints = {f"custom_{args.joint}": args.joint}
    else:
        test_joints = TEST_JOINTS

    print("=" * 60)
    print("Big Bertha Actuator Step-Response Test")
    print(f"Step amplitude: {STEP_AMPLITUDE} rad")
    print(f"Settle time: {SETTLE_TIME}s, Record time: {RECORD_TIME}s")
    print("=" * 60)

    all_metrics = []
    for group_name, joint_name in test_joints.items():
        print(f"\n--- Testing {group_name} ({joint_name}) ---")
        metrics = run_step_test(joint_name, headless=args.headless)
        all_metrics.append(metrics)
        print(f"  Rise time:     {metrics['rise_time_s']:.4f} s")
        print(f"  Overshoot:     {metrics['overshoot_pct']:.2f}%")
        print(f"  Settle time:   {metrics['settle_time_s']:.4f} s")
        print(f"  SS error:      {metrics['steady_state_error']:.4f} rad")
        print(f"  CSV saved to:  {metrics['csv_path']}")

    # Summary table
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"{'Group':<10} {'Rise (s)':<12} {'Overshoot%':<12} {'Settle (s)':<12} {'SS error':<12}")
    print("-" * 58)
    for m in all_metrics:
        print(f"{m['group']:<10} {m['rise_time_s']:<12.4f} {m['overshoot_pct']:<12.2f} {m['settle_time_s']:<12.4f} {m['steady_state_error']:<12.4f}")

    # Interpretation
    print("\nInterpretation:")
    for m in all_metrics:
        status = "OK"
        if m['overshoot_pct'] > 15:
            status = "UNDERDAMPED — increase kd"
        elif m['overshoot_pct'] < 2:
            status = "OVERDAMPED — decrease kd"
        elif m['rise_time_s'] > 0.1:
            status = "SLOW — increase kp"
        print(f"  {m['group']:>6} ({m['joint']}): ζ≈{max(0.0, 1 - m['overshoot_pct']/100):.2f}, {status}")


if __name__ == "__main__":
    main()
