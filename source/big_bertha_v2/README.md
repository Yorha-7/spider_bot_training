# Big Bertha v2

This is a work-in-progress copy of the Big Bertha walking task. Its runtime
robot asset is `assets/usd/big_bertha_v2/big_bertha_v2.usd`, converted from
`assets/URDF/big_bertha_v2/big_bertha_v2.urdf`. Install it with
`python -m pip install -e source/big_bertha_v2` in the Isaac Lab environment.
The RSL-RL task ID is `big_bertha_v2`.

Regenerate the USD after changing the URDF, using the installed Isaac Lab
environment:

```bash
python /path/to/IsaacLab/scripts/tools/convert_urdf.py \
  assets/URDF/big_bertha_v2/big_bertha_v2.urdf \
  assets/usd/big_bertha_v2/big_bertha_v2.usd \
  --joint-stiffness 0 --joint-damping 0 --headless
```

The zero drive gains match the prior direct-URDF loading behavior. Keep the
floating base and fixed-joint merging disabled. Commit the top-level USD and
all files under `assets/usd/big_bertha_v2/configuration/` together. The task
uses the same grid ground asset and dome light as v1. It overrides the robot's
USD visuals with a uniform silver `(0.7, 0.7, 0.7)` material; that runtime
color does not change the standalone USD or its source URDF.

The lower-leg collision meshes are generated from the CAD OBJ groups instead
of using one box around each whole lower-leg assembly. Regenerate them before
the USD when a lower-leg visual mesh or its URDF transform changes:

```bash
python scripts/build_big_bertha_v2_collisions.py
```

Use a fresh training run after changing collision geometry or foot-tip offsets;
existing checkpoints were trained with the previous contact physics and reward.

The policy observation has 56 values: the previous 52 values followed by four
binary contact switches, ordered right front, left front, left rear, right rear.
Each switch activates at a lower-leg contact force of at least 0.98 N. The
lower-leg bodies are the physical feet in the latest URDF; no synthetic shoe
links are added. Joint positions in the observation are the current URDF joint
angles, not offsets from the initial positions.

The reward pairs right front with left rear and left front with right rear,
half a cycle apart. During planar commands, the exact-pair bonus requires both
scheduled feet to swing, the other pair to support, and achieved velocity along
the command. A smaller alignment term gives partial credit when one scheduled
foot swings or one supporting foot is still planted. It requires at least one
scheduled swing foot to lift; standing on all four feet earns none. Its minimum
progress gate is 20%, so early steps can earn a small signal before the robot
moves. Both terms are zero for stop and pure-turn commands. The four clock
observation channels remain unchanged, but an older checkpoint will need
further training to learn the new gait objective.

The source URDF and converted USD use the mass and inertia estimates in
`assets/URDF/big_bertha_v2/mass_estimates.json`: uniform effective PLA density
of 248 kg/m³ (20% of 1240 kg/m³) and 55 g per MG995 servo. Other components,
including electronics and the battery, are treated as PLA. This simple estimate
does not account for solid walls and some source meshes are not watertight.

The initial pose puts all four CAD sole centres about 3 mm above the floor at
the 108 mm base-height target, with each sole within 2 degrees of level. With
10.0 stiffness, 0.5 damping, four position and one velocity solver iteration,
and per-iteration external-force updates, a ten-second clean, zero-action
simulation held the base between 102.5 and 106.0 mm, with less than 0.4 degrees
of tilt and no termination. This checks the reset pose under a fixed stop
command; it does not validate a walking policy or randomized conditions.

### Swing-clearance reward

The default training reward uses a fixed 10 mm clearance target. Each completed
swing is measured from that foot's last contact height, so it rewards lifting
the foot rather than merely keeping it airborne. Only swings lasting at least
`swing_min_duration` (0.10 seconds) can earn a touchdown reward. This tests
clearance only, not velocity tracking or overall gait quality.

The same fixed target is used by playback and does not change policy
observations.

### Walking reward shaping

- Height is a cost outside 108 ± 10 mm, with no cost inside the band. The
  excess is normalized by 20 mm and squared, with weight -2. Tilt penalties
  and fall termination remain active.
- Action-rate and slip penalties start at 25% strength. A 10-second exponential
  moving average of translational command-tracking quality drives the ramp:
  50% quality begins promotion and 90% reaches full strength. Quality is
  `clamp(1 - velocity_error / command_speed, 0, 1)`, averaged over environments
  with translational commands above 0.05 m/s. Standing cannot promote the ramp;
  its multiplier never decreases. This curriculum targets the forward-walking
  baseline; pure yaw commands do not advance it.
- Clearance has a per-event weight of 0.4, without timestep multiplication.
  One perfect foot touchdown earns 0.1 after the four-foot average (previously
  0.02 at 50 Hz). The Gaussian clearance sigma remains 20 mm. Short touchdowns
  still end their swing but earn nothing, avoiding delayed payment while planted.
- Reward contact detection switches on above 0.98 N and off at or below 0.70 N.
  The four binary observation switches retain their existing 0.98 N contract.
  Slip costs average squared speed beyond a 0.01 m/s tolerance over contacting
  feet, weighted by -20 and the movement curriculum multiplier. These are
  initial tuning values, not simulator-validated optimums.
- `foot_tip_offsets` now uses the centre of each CAD sole's lowest face in its
  lower-link frame. The earlier left-foot offsets were about 29 cm from their
  soles, so their slip and clearance rewards measured the wrong points. The
  new lower-link colliders use an arm hull and separate servo and foot boxes.
  The offsets remain sole proxies, not exact PhysX contact locations.

Training logs now include per-foot `Foot/drag_reward_*`,
`Foot/contact_fraction_*`, and `Foot/contact_speed_mps_*` for FR, FL, RL, RR.
The four drag-reward values sum to `Episode_Reward/foot_dragging` at episode
reset; they do not change the reward or observations.
`Gait/exact_pair_fraction` and `Gait/contact_alignment` report contact quality
over moving pair windows; `Gait/moving_pair_steps` shows their sample count.
The alignment metric measures contacts alone, independent of achieved speed.

Monitor `Curriculum/movement_penalty_multiplier` and
`Curriculum/movement_tracking_quality`. Like the clearance curriculum, movement
curriculum state is not saved in policy checkpoints. Disable adaptation for
fixed evaluation with `env.movement_penalty_curriculum=false` (full penalties).
To resume with a previously reached multiplier as the floor, override
`env.movement_penalty_initial` with the logged value. New training processes
otherwise start at 25%.
