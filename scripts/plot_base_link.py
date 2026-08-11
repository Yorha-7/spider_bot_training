#!/usr/bin/env python3
"""Top-down schematic: where base_link sat before the move, and where it sits now.

The robot itself does not move. Only the frame we call base_link does, so the two
panels show the same physical machine with the origin marked in different places.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# ROS REP-103: +x forward, +y left. Hip joint origins in the base_link frame.
OLD = {
    "FL (R110)": (0.129, 0.015),
    "BL (R113)": (-0.015, -0.002),
    "BR (R116)": (0.0027, -0.145),
    "FR (R119)": (0.145, -0.128),
}
CX, CY = 0.06489606, -0.06499993  # base_link inertial origin == COM
NEW = {k: (x - CX, y - CY) for k, (x, y) in OLD.items()}

INK, MUTED = "#1f2933", "#7b8794"
LEG, OLDC, NEWC = "#3d5a80", "#c1462f", "#2a7f62"

fig, axes = plt.subplots(1, 2, figsize=(12.4, 6.4))
fig.patch.set_facecolor("white")

for ax, (title, hips, origin_label, ocol, lbl_off) in zip(
    axes,
    [
        ("BEFORE  —  base_link 92 mm off centre", OLD, "base_link (old)", OLDC, (42, 46)),
        ("AFTER  —  base_link at the body centre", NEW, "base_link (new)", NEWC, (-8, -42)),
    ],
):
    # body outline through the four hips
    pts = list(hips.values()) + [list(hips.values())[0]]
    ax.plot([p[0] for p in pts], [p[1] for p in pts], "-", color=LEG, lw=2, alpha=0.35, zorder=1)

    for name, (x, y) in hips.items():
        ax.plot(x, y, "o", ms=13, color=LEG, zorder=3)
        ax.annotate(
            name,
            (x, y),
            textcoords="offset points",
            xytext=(0, 17),
            ha="center",
            fontsize=10.5,
            color=INK,
            weight="bold",
        )
        ax.annotate(
            f"({x:+.3f}, {y:+.3f})",
            (x, y),
            textcoords="offset points",
            xytext=(0, -26),
            ha="center",
            fontsize=8.5,
            color=MUTED,
        )

    # the frame origin is always (0,0) by definition of the frame
    ax.plot(0, 0, "X", ms=19, color=ocol, mew=2.5, zorder=5)
    ax.annotate(
        origin_label,
        (0, 0),
        textcoords="offset points",
        xytext=lbl_off,
        ha="center",
        fontsize=11,
        color=ocol,
        weight="bold",
    )

    cx = sum(p[0] for p in hips.values()) / 4
    cy = sum(p[1] for p in hips.values()) / 4
    if hips is OLD:
        # body centre, and the offset the move removes
        ax.plot(cx, cy, "+", ms=19, color=NEWC, mew=2.5, zorder=4)
        ax.annotate(
            "body centre",
            (cx, cy),
            textcoords="offset points",
            xytext=(14, -4),
            ha="left",
            fontsize=10,
            color=NEWC,
            weight="bold",
        )
        ax.annotate(
            "", xy=(cx, cy), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=OLDC, lw=2.2, ls=(0, (4, 2)))
        )
        ax.annotate(
            "92 mm",
            (cx / 2, cy / 2),
            textcoords="offset points",
            xytext=(20, -12),
            ha="left",
            fontsize=10.5,
            color=OLDC,
            weight="bold",
        )
    else:
        # base_link and the body centre now coincide, so no second marker
        ax.add_patch(Circle((0, 0), 0.102, fill=False, ec=MUTED, ls=":", lw=1.4))
        ax.annotate(
            "all hips at 101–103 mm radius",
            (0, 0.102),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=9.5,
            color=MUTED,
            style="italic",
        )

    ax.axhline(0, color=MUTED, lw=0.8, alpha=0.4)
    ax.axvline(0, color=MUTED, lw=0.8, alpha=0.4)
    ax.set_title(title, fontsize=12.5, color=INK, weight="bold", pad=16)
    ax.set_xlabel("+x  forward  (m)", fontsize=10, color=MUTED)
    ax.set_ylabel("+y  left  (m)", fontsize=10, color=MUTED)
    ax.set_xlim(-0.16, 0.24)
    ax.set_ylim(-0.22, 0.16)
    ax.set_aspect("equal")
    ax.grid(alpha=0.18, lw=0.7)
    for s in ax.spines.values():
        s.set_color(MUTED)
        s.set_alpha(0.4)
    ax.tick_params(colors=MUTED, labelsize=8.5)

fig.suptitle(
    "base_link moved to the body centre  —  the robot itself does not move, only the frame",
    fontsize=13.5,
    color=INK,
    weight="bold",
    y=0.98,
)
fig.text(
    0.5,
    0.022,
    "Yaw commands are tracked at base_link, so it is the point the robot turns about. "
    "It previously sat 15 mm from leg FL's hip.",
    ha="center",
    fontsize=10,
    color=MUTED,
)
fig.tight_layout(rect=[0, 0.075, 1, 0.94])
out = "/home/jjateen/Desktop/tmp/v200/base_link_move.png"
fig.savefig(out, dpi=155, facecolor="white")
print("wrote", out)
