"""Tensor helpers for contact-aware walking rewards (no simulator imports)."""

import torch


def filtered_contact(force, previous_contact, on_threshold, off_threshold):
    """Apply separate contact entry and exit force thresholds to reduce flicker."""
    return torch.where(previous_contact, force > off_threshold, force > on_threshold)


def slip_cost(tip_speed, contact, tolerance):
    """Average squared excess tip speed across feet that are in contact."""
    return (contact.float() * torch.relu(tip_speed - tolerance).square()).mean(dim=1)


def height_band_cost(height, target, tolerance, error_scale):
    """Return scaled squared height error outside the target tolerance band."""
    return (torch.relu((height - target).abs() - tolerance) / error_scale).square()


def diagonal_pair_terms(contact, phase, offsets, commands, velocity, stance_fraction, minimum_progress_gate=0.2):
    """Return exact and partial diagonal rewards with their contact diagnostics.

    Feet are ordered FR, FL, RL, RR. A correct swing has FR+RL airborne or
    FL+RR airborne, with the other diagonal supporting the robot. The phase
    schedule leaves an all-stance interval between pair swings.
    """
    foot_phase = (phase.unsqueeze(1) + offsets.unsqueeze(0)) % 1.0
    expected_air = foot_phase >= stance_fraction
    exact_pattern = (contact == ~expected_air).all(dim=1)
    pair_window = expected_air.sum(dim=1) == 2
    swing_match = ((~contact) & expected_air).float().sum(dim=1) / 2.0
    support_match = (contact & ~expected_air).float().sum(dim=1) / 2.0
    alignment = swing_match * support_match * pair_window.float()

    command_speed_sq = torch.sum(commands.square(), dim=1)
    progress = torch.sum(commands * velocity, dim=1) / command_speed_sq.clamp_min(0.05**2)
    moving = command_speed_sq > 0.05**2
    progress = progress.clamp(0.0, 1.0)
    exact_reward = exact_pattern.float() * pair_window.float() * progress * moving.float()
    partial_reward = alignment * (minimum_progress_gate + (1.0 - minimum_progress_gate) * progress) * moving.float()
    return exact_reward, partial_reward, pair_window, exact_pattern, alignment


def diagonal_pair_reward(contact, phase, offsets, commands, velocity, stance_fraction):
    """Existing exact-pattern bonus, retained for callers outside the environment."""
    return diagonal_pair_terms(contact, phase, offsets, commands, velocity, stance_fraction)[0]


def update_movement_curriculum(ema, multiplier, commands, velocity, dt, time_constant, initial, low, high):
    """Ramp only from translational tracking; standing cannot promote it.

    Pool moving environments. Relative velocity error distinguishes standing
    still from tracking even at small command magnitudes. The multiplier never
    decreases, so a policy cannot relax penalties by deliberately tracking badly.
    """
    speed = commands.norm(dim=1)
    moving = speed > 0.05
    quality = (1.0 - (commands - velocity).norm(dim=1) / speed.clamp_min(0.05)).clamp(0.0, 1.0)
    average = (quality * moving).sum() / moving.sum().clamp_min(1)
    alpha = 1.0 - torch.exp(ema.new_tensor(-dt / time_constant))
    ema = torch.where(moving.any(), ema + alpha * (average - ema), ema)
    progress = ((ema - low) / (high - low)).clamp(0.0, 1.0)
    multiplier = torch.maximum(multiplier, initial + (1.0 - initial) * progress)
    return ema, multiplier
