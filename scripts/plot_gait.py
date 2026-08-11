#!/usr/bin/env python3
"""Gait diagnostics from a BB_GAIT_DUMP rollout.

Three plots that answer questions no reward curve can:

  1. Hildebrand footfall diagram - one bar per leg, filled during stance. Reads
     out gait type, duty factor and the phase offsets between legs, and shows
     directly whether a leg ever leaves the ground. This is the plot that would
     have settled the hardware "legs are not lifting" question in one look.
  2. Foot trajectory in the body frame - the arc each foot traces, so swing
     height and stride length are measurable rather than inferred.
  3. Support polygon area over time - how much base of support the gait keeps.
     Meaningful now that base_link actually sits at the body centre.

Inputs come from the same contact sensor and FK the reward terms use, so the
plots and the reward agree by construction. The dump also works for Gazebo and
hardware rollouts, which makes the same figure comparable across all three.

Usage: plot_gait.py <dump.npz> <out.png> [title]
"""

from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# _feet_ids order, derived from the URDF rather than from the env comment.
# find_bodies([arm_c_1_1..arm_c_4_1]) resolves through Revolute_113/116/119/110,
# which sit back-left / back-right / front-right / front-left. The comment in
# big_bertha_env.py claims FR/FL/RL/RR and is wrong; so were the leg labels in
# symmetry.py and legged_odometry.yaml, each differently.
LEGS = ["BL", "BR", "FR", "FL"]
INK, MUTED = "#1f2933", "#7b8794"
STANCE, SWING = "#3d5a80", "#e8edf2"
ACCENT = "#c1462f"
CONTACT_N = 1.0  # newtons; above this the foot is loaded


def main(dump: str, out: str, title: str = "") -> None:
    d = np.load(dump)
    contact, tip, dt = d["contact"], d["tip_b"], float(d["dt"])
    cmd = d["cmd"]
    T = contact.shape[0]
    t = np.arange(T) * dt
    stance = contact > CONTACT_N

    fig = plt.figure(figsize=(13.5, 8.2))
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.15], hspace=0.34, wspace=0.24)

    # ---- 1. Hildebrand footfall -------------------------------------------
    ax = fig.add_subplot(gs[0, :])
    for i, leg in enumerate(LEGS):
        y = len(LEGS) - 1 - i
        ax.broken_barh(
            [(t[a], t[b] - t[a]) for a, b in _runs(stance[:, i])],
            (y - 0.34, 0.68),
            facecolors=STANCE,
        )
        duty = stance[:, i].mean()
        ax.text(t[-1] * 1.012, y, f"duty {duty:.2f}", va="center", fontsize=9.5, color=MUTED)
    ax.set_yticks(range(len(LEGS)))
    ax.set_yticklabels(LEGS[::-1], fontsize=11, color=INK)
    ax.set_xlim(0, t[-1] * 1.10)
    ax.set_xlabel("time (s)", fontsize=10, color=MUTED)
    ax.set_title(
        "Footfall diagram   (filled = stance, blank = swing)", fontsize=12, color=INK, weight="bold", loc="left"
    )
    _clean(ax)

    # ---- 2. foot trajectories in body frame -------------------------------
    ax = fig.add_subplot(gs[1, 0])
    for i, leg in enumerate(LEGS):
        ax.plot(tip[:, i, 0], tip[:, i, 2], lw=1.1, alpha=0.85, label=leg)
    ax.set_xlabel("x forward (m)", fontsize=10, color=MUTED)
    ax.set_ylabel("z height (m)", fontsize=10, color=MUTED)
    ax.set_title("Foot tip path, body frame", fontsize=12, color=INK, weight="bold", loc="left")
    ax.legend(fontsize=9, frameon=False, ncol=4, loc="upper center")
    _clean(ax)

    # ---- 3. support polygon area ------------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    area = np.array([_poly_area(tip[k, stance[k], :2]) for k in range(T)])
    ax.plot(t, area * 1e4, lw=1.4, color=ACCENT)
    ax.axhline(0, color=MUTED, lw=0.8, alpha=0.5)
    ax.set_xlabel("time (s)", fontsize=10, color=MUTED)
    ax.set_ylabel("area (cm$^2$)", fontsize=10, color=MUTED)
    nfeet = stance.sum(axis=1)
    ax.set_title(
        f"Support polygon   (mean {area.mean() * 1e4:.0f} cm2, {100 * (nfeet < 3).mean():.0f}% of time under 3 feet)",
        fontsize=12,
        color=INK,
        weight="bold",
        loc="left",
    )
    _clean(ax)

    head = title or f"cmd vx={cmd[0]:+.2f}  vy={cmd[1]:+.2f}  wz={cmd[2]:+.2f}"
    fig.suptitle(head, fontsize=13.5, color=INK, weight="bold", y=0.985)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    fig.savefig(out, dpi=150, facecolor="white")
    print(f"wrote {out}")
    nfeet = stance.sum(axis=1)
    print(f"  mean feet loaded {nfeet.mean():.2f}   under 3 feet {100 * (nfeet < 3).mean():.0f}% of the time")
    print(f"  peak contact force {contact.max():.0f} N   (body weight ~12.6 N)")
    for i, leg in enumerate(LEGS):
        print(
            f"  {leg}: duty {stance[:, i].mean():.3f}  "
            f"swing height {tip[:, i, 2].max() - tip[:, i, 2].min():.4f} m  "
            f"peak {contact[:, i].max():.0f} N"
        )


def _runs(mask):
    """Contiguous True runs as (start, end) index pairs."""
    idx = np.flatnonzero(np.diff(np.concatenate(([0], mask.view(np.int8), [0]))))
    return list(zip(idx[::2], np.minimum(idx[1::2], len(mask) - 1)))


def _poly_area(pts):
    """Shoelace area of the convex hull of the loaded feet."""
    if len(pts) < 3:
        return 0.0
    c = pts.mean(axis=0)
    p = pts[np.argsort(np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0]))]
    return 0.5 * abs(np.dot(p[:, 0], np.roll(p[:, 1], -1)) - np.dot(p[:, 1], np.roll(p[:, 0], -1)))


def _clean(ax):
    ax.grid(alpha=0.18, lw=0.7)
    for s in ax.spines.values():
        s.set_color(MUTED)
        s.set_alpha(0.4)
    ax.tick_params(colors=MUTED, labelsize=8.5)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "")
