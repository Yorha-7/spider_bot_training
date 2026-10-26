#!/usr/bin/env python3
"""Publication figures for the Big Bertha training campaign.

Training ran as a chain of resumed runs sharing one iteration counter, so the
lineage is rebuilt by walking back from the final policy: each run starts at
the iteration its parent stopped at. Runs off that chain are abandoned
branches and are excluded.

Outputs (PNG, 300 dpi) into --outdir:
  fig1_learning_curve.png   return + episode length over the whole campaign
  fig2_reward_terms.png     per-term reward decomposition
  fig3_optimization.png     losses, learning rate, action-noise scale
  fig4_finetune.png         v1.0.0 -> v1.1.0 fine-tuning phase
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator  # noqa: E402

# Paper styling: serif, hairline axes, no chartjunk.
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif'],
    'font.size': 8,
    'axes.labelsize': 8,
    'axes.titlesize': 8.5,
    'legend.fontsize': 6.5,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'lines.linewidth': 1.0,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
})
ACCENT = '#0072B2'
ACCENT2 = '#D55E00'
GREY = '#999999'


def kfmt(v, _pos=None):
    """Format iteration ticks as 0, 20k, 40k ... (raw counts collide)."""
    return '0' if v == 0 else f'{v / 1000:g}k'


def load(evfile, tags):
    """Read the requested scalar tags from one event file."""
    ea = EventAccumulator(evfile, size_guidance={'scalars': 0})
    ea.Reload()
    have = set(ea.Tags()['scalars'])
    out = {}
    for t in tags:
        if t in have:
            s = ea.Scalars(t)
            out[t] = (np.array([x.step for x in s]), np.array([x.value for x in s]))
    return out


def build_chain(index, final_end, tol=300):
    """Walk back from the final run to the start of training.

    A resumed run begins where its parent stopped, but not always exactly: a
    resume can pick an earlier checkpoint than the parent's last logged
    iteration (e.g. 106900 after a parent ending at 106901), so the match is
    nearest-end-within-tol rather than equality. Among candidates prefer the
    longest run, which is the real trunk rather than a short probe.
    """
    runs = [r for r in index if r.get('n')]
    chain = [max((r for r in runs if r['end'] == final_end), key=lambda r: r['n'])]
    seen = {chain[0]['dir']}
    while True:
        cur = chain[-1]
        parents = [p for p in runs
                   if p['dir'] not in seen
                   and p['start'] < cur['start']
                   and abs(p['end'] - cur['start']) <= tol]
        if not parents:
            break
        nxt = max(parents, key=lambda p: p['n'])
        chain.append(nxt)
        seen.add(nxt['dir'])
    return list(reversed(chain))


def smooth(y, frac=0.006):
    """Centred running mean with a shrinking window at the edges.

    np.convolve(mode='same') zero-pads, which drags the curve toward zero at
    both ends and invents a collapse in the final iterations that never
    happened. Averaging only over samples that exist keeps the endpoints
    honest.
    """
    n = len(y)
    w = max(1, int(n * frac))
    if w < 2 or n < 3:
        return y
    half = w // 2
    c = np.cumsum(np.insert(np.asarray(y, dtype=float), 0, 0.0))
    idx = np.arange(n)
    lo = np.maximum(idx - half, 0)
    hi = np.minimum(idx + half + 1, n)
    return (c[hi] - c[lo]) / (hi - lo)


def gather(chain, tags):
    """Concatenate a tag across the chain, in iteration order."""
    series = {t: ([], []) for t in tags}
    for run in chain:
        ev = sorted(glob.glob(run['dir'] + 'events.out.tfevents*'))[0]
        d = load(ev, tags)
        for t, (x, y) in d.items():
            series[t][0].append(x)
            series[t][1].append(y)
    out = {}
    for t, (xs, ys) in series.items():
        if not xs:
            continue
        x = np.concatenate(xs)
        y = np.concatenate(ys)
        o = np.argsort(x, kind='stable')
        out[t] = (x[o], y[o])
    return out


def fig_learning(series, bounds, outdir):
    fig, axes = plt.subplots(2, 1, figsize=(3.5, 3.4), sharex=True)
    for ax, tag, lab in [
            (axes[0], 'Train/mean_reward', 'Mean episode return'),
            (axes[1], 'Train/mean_episode_length', 'Episode length (steps)')]:
        if tag not in series:
            continue
        x, y = series[tag]
        ax.plot(x, y, color=GREY, lw=0.3, alpha=0.5)
        ax.plot(x, smooth(y), color=ACCENT, lw=1.0)
        ax.set_ylabel(lab)
        for b in bounds[1:]:
            ax.axvline(b, color=GREY, lw=0.4, ls=':', alpha=0.7)
    axes[1].set_xlabel('PPO iteration')
    axes[1].xaxis.set_major_formatter(plt.FuncFormatter(kfmt))
    axes[0].plot([], [], color=GREY, lw=0.4, ls=':', label='run resume')
    axes[0].legend(frameon=False, loc='lower right')
    fig.align_ylabels(axes)
    fig.savefig(os.path.join(outdir, 'fig1_learning_curve.png'))
    plt.close(fig)


def fig_terms(chain, outdir):
    tags = [f'Episode_Reward/{k}' for k in [
        'track_lin_vel_xy_exp', 'forward_progress', 'track_ang_vel_z_exp', 'raibert',
        'foot_clearance', 'crawl_gait', 'joint_deviation', 'flat_orientation_l2',
        'dof_acc_l2', 'action_rate_l2', 'dof_torques_l2', 'base_height']]
    s = gather(chain, tags)
    fig, axes = plt.subplots(4, 3, figsize=(7.0, 5.2), sharex=True)
    for ax, t in zip(axes.ravel(), tags):
        name = t.split('/')[1]
        if t not in s:
            ax.set_visible(False)
            continue
        x, y = s[t]
        pos = y.mean() >= 0
        ax.plot(x, smooth(y), color=ACCENT if pos else ACCENT2, lw=0.9)
        ax.axhline(0, color=GREY, lw=0.4)
        ax.set_title(name.replace('_', r'\_') if False else name, pad=2)
        ax.tick_params(labelsize=6)
    for ax in axes[-1]:
        ax.set_xlabel('PPO iteration')
        ax.xaxis.set_major_formatter(plt.FuncFormatter(kfmt))
    fig.suptitle('Reward-term contributions (blue: incentive, orange: penalty)',
                 fontsize=8.5, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(os.path.join(outdir, 'fig2_reward_terms.png'))
    plt.close(fig)


def fig_optim(series, bounds, outdir):
    panels = [('Loss/surrogate', 'Surrogate loss'), ('Loss/value', 'Value loss'),
              ('Loss/entropy', 'Entropy'), ('Loss/learning_rate', 'Learning rate'),
              ('Policy/mean_std', r'Action noise $\sigma$'),
              ('Perf/total_fps', 'Throughput (FPS)')]
    fig, axes = plt.subplots(3, 2, figsize=(7.0, 4.4), sharex=True)
    for ax, (tag, lab) in zip(axes.ravel(), panels):
        if tag not in series:
            ax.set_visible(False)
            continue
        x, y = series[tag]
        ax.plot(x, smooth(y), color=ACCENT, lw=0.9)
        ax.set_ylabel(lab)
        # mean_std spans ~0.5 to ~2400: linear hides everything.
        if tag in ('Policy/mean_std', 'Loss/learning_rate', 'Loss/value') and y.min() > 0:
            ax.set_yscale('log')
        for b in bounds[1:]:
            ax.axvline(b, color=GREY, lw=0.3, ls=':', alpha=0.6)
    for ax in axes[-1]:
        ax.set_xlabel('PPO iteration')
        ax.xaxis.set_major_formatter(plt.FuncFormatter(kfmt))
    fig.align_ylabels(axes)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'fig3_optimization.png'))
    plt.close(fig)


# Post-hoc evaluation, both policies measured on the SAME environment after
# training (Gazebo, MG995 torque-speed plant, 20 s per condition; Isaac from
# BB_EVAL_CLEAN, 64 envs). Training return cannot be used for this comparison:
# the reward function was revised during the fine-tune, so returns either side
# of the split score different objectives.
EVAL = {
    'Gazebo (deployment plant)': {
        r'$v_x$ 0.12': (0.128, 0.135),
        r'$v_x$ 0.30': (0.180, 0.200),
        r'$\omega_z$ 0.5': (0.298, 0.275),
    },
    'Isaac (training plant)': {
        r'$v_x$ 0.12': (0.159, 0.157),
        r'$v_x$ 0.30': (0.271, 0.268),
        r'$\omega_z$ 0.5': (0.422, 0.424),
    },
}


def fig_finetune(chain, split, outdir):
    """Plot the fine-tune: noise reset during it, measured effect after it."""
    s = gather(chain, ['Policy/mean_std'])
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.2))

    ax = axes[0]
    if 'Policy/mean_std' in s:
        x, y = s['Policy/mean_std']
        m = x >= split - 4000
        ax.plot(x[m & (x <= split)], y[m & (x <= split)], color=GREY, lw=1.0,
                label='v1.0.0')
        ax.plot(x[x >= split], y[x >= split], color=ACCENT, lw=1.0,
                label='v1.1.0')
        ax.axvline(split, color=ACCENT2, lw=0.7, ls='--')
        ax.set_yscale('log')
        ax.set_ylabel(r'Action noise $\sigma$')
        ax.set_xlabel('PPO iteration')
        ax.xaxis.set_major_formatter(plt.FuncFormatter(kfmt))
        ax.legend(frameon=False, loc='center left')
        ax.set_title('Exploration noise reset', pad=3)

    for ax, (title, metrics) in zip(axes[1:], EVAL.items()):
        labels = list(metrics)
        old = [metrics[k][0] for k in labels]
        new = [metrics[k][1] for k in labels]
        pos = np.arange(len(labels))
        ax.bar(pos - 0.19, old, 0.36, color=GREY, label='v1.0.0')
        ax.bar(pos + 0.19, new, 0.36, color=ACCENT, label='v1.1.0')
        for i, (a, b) in enumerate(zip(old, new)):
            ax.text(i + 0.19, b, f'{100 * (b - a) / a:+.0f}%', ha='center',
                    va='bottom', fontsize=6)
        ax.set_xticks(pos)
        ax.set_xticklabels(labels, fontsize=6.5)
        ax.set_ylabel('measured (m/s, rad/s)')
        ax.set_ylim(0, max(old + new) * 1.40)
        ax.set_title(title, pad=3)
        ax.legend(frameon=False, fontsize=6, ncol=2, loc='upper left')
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'fig4_finetune.png'))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--index', default=os.path.expanduser('~/Desktop/tmp/runs_index.json'))
    ap.add_argument('--final-end', type=int, default=139897, help='last iteration of v1.1.0')
    ap.add_argument('--split', type=int, default=137398, help='v1.0.0 -> v1.1.0 branch point')
    ap.add_argument('--outdir', default=os.path.expanduser('~/Desktop/tmp/figs'))
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    index = json.load(open(a.index))
    chain = build_chain(index, a.final_end)
    bounds = [r['start'] for r in chain]
    total = sum(r['n'] for r in chain)
    print(f'chain: {len(chain)} runs, {chain[0]["start"]} -> {chain[-1]["end"]} '
          f'({total} logged iterations)')
    for r in chain:
        print(f'  {os.path.basename(r["dir"].rstrip("/")):22s} '
              f'{r["start"]:>7d}->{r["end"]:>7d}  n={r["n"]}')

    core = gather(chain, ['Train/mean_reward', 'Train/mean_episode_length',
                          'Loss/surrogate', 'Loss/value', 'Loss/entropy',
                          'Loss/learning_rate', 'Policy/mean_std', 'Perf/total_fps'])
    fig_learning(core, bounds, a.outdir)
    fig_optim(core, bounds, a.outdir)
    fig_terms(chain, a.outdir)
    fig_finetune(chain, a.split, a.outdir)
    print('wrote:', ', '.join(sorted(os.listdir(a.outdir))))


if __name__ == '__main__':
    main()
