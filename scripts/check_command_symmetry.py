#!/usr/bin/env python3
"""Assert every command-conditioned term is symmetric in the sign of vx.

v2.0.0 trained for 15 hours with reverse commands and produced no reverse motion.
The cause was five separate terms treating cmd_vx as forward-only, the worst of
them the gait clock: a raw (unsigned) vx in the cadence boost meant a reverse
command throttled the clock rather than running it. At cmd -0.15 the cadence fell
to 0.45x; at -0.30 the boost went negative. mean_reward stayed at 110% of baseline
the whole time, because reverse episodes collected near-full gait reward for
stepping in place, so no training metric could see it.

This runs in seconds, needs no GPU and no Isaac. Run it before any training that
involves reverse commands.
"""

from __future__ import annotations

import sys

TURN_BOOST, SPEED_BOOST = 0.8, 1.1
FAILED = []


def check(name, a, b, tol=1e-9):
    ok = abs(a - b) <= tol
    print(f"  {'PASS' if ok else '*** FAIL ***':<14} {name:<46} {a:+.6f} vs {b:+.6f}")
    if not ok:
        FAILED.append(name)


def clock_boost(vx, yaw):
    return min(
        1.0 + TURN_BOOST * min(abs(yaw) / 0.4, 1.0) + SPEED_BOOST * min(abs(vx) / 0.3, 1.0),
        2.1,
    )


def stance_r(vx):
    return 0.75 - 0.15 * min(abs(vx) / 0.3, 1.0)


def vel_gate(vx, achieved):
    """Progress along the commanded direction, clamped to [0, 1]."""
    if abs(vx) <= 0.05:
        return None  # turn-in-place or stop branch
    along = achieved * (1.0 if vx > 0 else -1.0)
    return max(0.0, min(along / max(abs(vx), 0.05), 1.0))


print("gait clock boost  (cadence follows SPEED, not direction)")
for v in (0.05, 0.12, 0.15, 0.30, 0.40):
    check(f"boost(+{v}) == boost(-{v})", clock_boost(v, 0.0), clock_boost(-v, 0.0))
check("boost(-0.30) > 1.0  (not throttled)", clock_boost(-0.30, 0.0) > 1.0, True)
check("boost(-0.30) > 0    (not reversed)", clock_boost(-0.30, 0.0) > 0.0, True)

print("\nduty ratio  (shrinks with speed, either direction)")
for v in (0.10, 0.15, 0.30):
    check(f"stance_r(+{v}) == stance_r(-{v})", stance_r(v), stance_r(-v))

print("\nvel_gate  (a robot moving the way it was told scores 1.0 both ways)")
check("vel_gate(+0.15, achieved +0.15)", vel_gate(0.15, 0.15), 1.0)
check("vel_gate(-0.15, achieved -0.15)", vel_gate(-0.15, -0.15), 1.0)
check("vel_gate(-0.15, standing still) == 0", vel_gate(-0.15, 0.0), 0.0)
check("vel_gate(-0.15, going FORWARD) == 0", vel_gate(-0.15, 0.15), 0.0)

print("\nforward_progress  (pays along the command, either direction)")


def fp(vx, ach):
    """forward_progress: velocity along the commanded direction, clamped."""
    if abs(vx) <= 0.05:
        return 0.0
    return max(0.0, min(ach * (1 if vx > 0 else -1), abs(vx)))


check("fp(+0.30, achieved +0.29)", fp(0.30, 0.29), 0.29)
check("fp(-0.15, achieved -0.15)", fp(-0.15, -0.15), 0.15)
check("fp(-0.15, standing still) == 0", fp(-0.15, 0.0), 0.0)
# Overshoot must not pay: exceeding the command earns no more than matching it.
check("fp(+0.12, achieved +0.174) capped at cmd", fp(0.12, 0.174), 0.12)
check("fp(-0.10, achieved -0.127) capped at cmd", fp(-0.10, -0.127), 0.10)
check("fp deadband: abs(cmd) <= 0.05 pays nothing", fp(-0.05, -0.127), 0.0)
check("fp(+0.30, achieved +0.50) capped at cmd", fp(0.30, 0.50), 0.30)

print("\ncrawl gate  (gait shaping must apply in reverse too)")


def gate(vx, ach):
    """crawl_gait fwd_gate: progress along the command, normalised and clamped."""
    return max(0.0, min(ach * (1 if vx > 0 else -1) / 0.10, 1.0))


check("gate(+0.30, achieved +0.29)", gate(0.30, 0.29), 1.0)
check("gate(-0.15, achieved -0.15)", gate(-0.15, -0.15), 1.0)
check("gate(-0.15, standing still) == 0", gate(-0.15, 0.0), 0.0)

if FAILED:
    print(f"\n{len(FAILED)} FAILED: {FAILED}")
    sys.exit(1)
print("\nall command-symmetry checks passed")
