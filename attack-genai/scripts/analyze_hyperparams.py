#!/usr/bin/env python
"""Parse training logs and generate hyperparameter analysis plots."""
import re
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict
from scipy.ndimage import uniform_filter1d

LOG_DIR = "/home/taikun/rl-attack/rl_atk/attack-genai/trained_attacker"
OUT_DIR = "/home/taikun/rl-attack/rl_atk/attack-genai/analysis_plots"
os.makedirs(OUT_DIR, exist_ok=True)

ALPHAS = [0.3, 0.5, 0.7]
MASKS = [3, 5, 10]
COLORS = {3: '#1f77b4', 5: '#ff7f0e', 10: '#2ca02c'}
LINESTYLES = {0.3: '-', 0.5: '--', 0.7: ':'}

def parse_log(filepath):
    steps, val_accs, losses = [], [], []
    reward_steps, totals, advs, sems = [], [], [], []
    best_acc, best_step, best_path = None, None, None

    step_re = re.compile(r'Step: (\d+), Training Loss: ([\d.]+), Validation Accuracy: ([\d.]+)')
    reward_re = re.compile(r'Mean rewards \(last \d+ steps\): total=([\d.]+), adv=([\d.]+), sem=([\d.]+)')
    best_re = re.compile(r'Best model saved at step (\d+) with accuracy: ([\d.]+), path: (.+)')

    with open(filepath, 'r') as f:
        for line in f:
            m = step_re.search(line)
            if m:
                steps.append(int(m.group(1)))
                losses.append(float(m.group(2)))
                val_accs.append(float(m.group(3)))

            m = reward_re.search(line)
            if m:
                reward_steps.append(steps[-1] if steps else 0)
                totals.append(float(m.group(1)))
                advs.append(float(m.group(2)))
                sems.append(float(m.group(3)))

            m = best_re.search(line)
            if m:
                acc = float(m.group(2))
                if best_acc is None or acc < best_acc:
                    best_acc = acc
                    best_step = int(m.group(1))
                    best_path = m.group(3)

    return {
        'steps': np.array(steps), 'val_accs': np.array(val_accs), 'losses': np.array(losses),
        'reward_steps': np.array(reward_steps), 'totals': np.array(totals),
        'advs': np.array(advs), 'sems': np.array(sems),
        'best_acc': best_acc, 'best_step': best_step, 'best_path': best_path,
    }

# Parse all logs
data = {}
for alpha in ALPHAS:
    for masks in MASKS:
        path = os.path.join(LOG_DIR, f"train_log_alpha{alpha}_masks{masks}.txt")
        if os.path.exists(path):
            data[(alpha, masks)] = parse_log(path)
            print(f"α={alpha}, masks={masks}: {len(data[(alpha, masks)]['steps'])} eval points, "
                  f"best_acc={data[(alpha, masks)]['best_acc']}, step={data[(alpha, masks)]['best_step']}")

SMOOTH_WINDOW = 25

metric_defs = [
    ('val_accs', 'steps', 'Validation Accuracy (↓ better)'),
    ('advs', 'reward_steps', 'Adversarial Reward (↑ better)'),
    ('sems', 'reward_steps', 'Semantic Reward (↑ better)'),
    ('losses', 'steps', 'Training Loss (↓ better)'),
]

ALPHA_COLORS = {0.3: '#1f77b4', 0.5: '#ff7f0e', 0.7: '#2ca02c'}
MASK_COLORS = {3: '#1f77b4', 5: '#ff7f0e', 10: '#2ca02c'}

# Compute shared y-axis limits per metric across ALL configs
Y_LIMITS = {}
for metric_key, step_key, title in metric_defs:
    all_vals = []
    for key, d in data.items():
        vals = d[metric_key]
        if len(vals) > 0:
            smoothed = uniform_filter1d(vals.astype(float), size=SMOOTH_WINDOW)
            all_vals.extend(smoothed.tolist())
    if all_vals:
        ymin, ymax = min(all_vals), max(all_vals)
        margin = (ymax - ymin) * 0.08
        Y_LIMITS[metric_key] = (ymin - margin, ymax + margin)

X_MAX = 10000  # shared x-axis limit

def plot_grid(fig_path, fig_title, row_values, row_labels, row_param_name,
              line_values, line_colors, line_label_fmt, line_param_name):
    fig, axes = plt.subplots(3, 4, figsize=(18, 12), sharey=False)
    fig.suptitle(fig_title, fontsize=14, fontweight='bold', y=0.98)

    for row_idx, row_val in enumerate(row_values):
        for col_idx, (metric_key, step_key, title) in enumerate(metric_defs):
            ax = axes[row_idx, col_idx]
            for line_val in line_values:
                if row_param_name == 'masks':
                    key = (line_val, row_val)
                else:
                    key = (row_val, line_val)
                if key not in data:
                    continue
                d = data[key]
                x = d[step_key]
                y = d[metric_key]
                if len(x) == 0:
                    continue
                y_smooth = uniform_filter1d(y.astype(float), size=SMOOTH_WINDOW)
                ax.plot(x, y, color=line_colors[line_val], alpha=0.08, linewidth=0.5)
                ax.plot(x, y_smooth, color=line_colors[line_val], linewidth=2.5,
                        alpha=0.9, label=line_label_fmt.format(line_val))

            # Shared axes
            ax.set_xlim(0, X_MAX)
            ylo, yhi = Y_LIMITS[metric_key]
            ax.set_ylim(ylo, yhi)
            from matplotlib.ticker import MultipleLocator
            ax.yaxis.set_major_locator(MultipleLocator(0.05))
            if row_idx == 0:
                ax.set_title(title, fontsize=11)
            if row_idx == 2:
                ax.set_xlabel('Step', fontsize=10)
            else:
                ax.set_xticklabels([])
            if col_idx == 0:
                ax.set_ylabel(row_labels[row_idx], fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            if row_idx == 0 and col_idx == 0:
                ax.legend(fontsize=9, loc='best')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(OUT_DIR, fig_path), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {fig_path}")

# ============================================================
# FIGURE 1a: Effect of Alpha (one row per mask count)
# ============================================================
plot_grid(
    fig_path='1a_alpha_effect_trajectories.png',
    fig_title='Effect of α (Holding Masks Fixed) — Smoothed Trajectories',
    row_values=MASKS,
    row_labels=[f'Masks = {m}' for m in MASKS],
    row_param_name='masks',
    line_values=ALPHAS,
    line_colors=ALPHA_COLORS,
    line_label_fmt='α={}',
    line_param_name='alpha',
)

# ============================================================
# FIGURE 1b: Effect of Masks (one row per alpha)
# ============================================================
plot_grid(
    fig_path='1b_masks_effect_trajectories.png',
    fig_title='Effect of Mask Count (Holding α Fixed) — Smoothed Trajectories',
    row_values=ALPHAS,
    row_labels=[f'α = {a}' for a in ALPHAS],
    row_param_name='alpha',
    line_values=MASKS,
    line_colors=MASK_COLORS,
    line_label_fmt='masks={}',
    line_param_name='masks',
)

# ============================================================
# FIGURE 2: Heatmaps of final metrics
# ============================================================
fig, axes = plt.subplots(1, 4, figsize=(18, 4))
fig.suptitle('Final Metrics Heatmap (α × Masks)', fontsize=14, fontweight='bold')

heatmap_metrics = [
    ('Best Val Acc (↓)', lambda d: d['best_acc'] if d['best_acc'] else d['val_accs'].min(), 'RdYlGn'),
    ('Final Adv Reward (↑)', lambda d: d['advs'][-1] if len(d['advs']) > 0 else 0, 'RdYlGn_r'),
    ('Final Sem Reward (↑)', lambda d: d['sems'][-1] if len(d['sems']) > 0 else 0, 'RdYlGn_r'),
    ('Final Loss (↓)', lambda d: d['losses'][-1] if len(d['losses']) > 0 else 0, 'RdYlGn'),
]

for idx, (title, extractor, cmap) in enumerate(heatmap_metrics):
    ax = axes[idx]
    grid = np.full((len(ALPHAS), len(MASKS)), np.nan)
    for i, alpha in enumerate(ALPHAS):
        for j, masks in enumerate(MASKS):
            if (alpha, masks) in data:
                grid[i, j] = extractor(data[(alpha, masks)])

    im = ax.imshow(grid, cmap=cmap, aspect='auto')
    ax.set_xticks(range(len(MASKS)))
    ax.set_xticklabels(MASKS)
    ax.set_yticks(range(len(ALPHAS)))
    ax.set_yticklabels(ALPHAS)
    ax.set_xlabel('Masks')
    ax.set_ylabel('Alpha')
    ax.set_title(title, fontsize=10)

    for i in range(len(ALPHAS)):
        for j in range(len(MASKS)):
            if not np.isnan(grid[i, j]):
                ax.text(j, i, f'{grid[i, j]:.4f}', ha='center', va='center', fontsize=9, fontweight='bold')

    fig.colorbar(im, ax=ax, shrink=0.8)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '2_heatmaps.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved 2_heatmaps.png")

# ============================================================
# FIGURE 3: Effect of Alpha (holding masks fixed)
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
fig.suptitle('Effect of α (Adv/Sem Tradeoff Weight) — Holding Masks Fixed', fontsize=14, fontweight='bold')

for idx, masks in enumerate(MASKS):
    ax = axes[idx]
    alphas_vals = []
    best_accs = []
    final_advs = []
    final_sems = []
    for alpha in ALPHAS:
        if (alpha, masks) in data:
            d = data[(alpha, masks)]
            alphas_vals.append(alpha)
            best_accs.append(d['best_acc'] if d['best_acc'] else d['val_accs'].min())
            final_advs.append(d['advs'][-1] if len(d['advs']) > 0 else 0)
            final_sems.append(d['sems'][-1] if len(d['sems']) > 0 else 0)

    ax2 = ax.twinx()
    l1 = ax.plot(alphas_vals, best_accs, 'o-', color='red', linewidth=2, markersize=8, label='Best Val Acc (↓)')
    l2 = ax2.plot(alphas_vals, final_advs, 's--', color='blue', linewidth=2, markersize=8, label='Final Adv Rwd (↑)')
    l3 = ax2.plot(alphas_vals, final_sems, '^:', color='green', linewidth=2, markersize=8, label='Final Sem Rwd (↑)')

    ax.set_xlabel('Alpha (α)')
    ax.set_ylabel('Val Accuracy', color='red')
    ax2.set_ylabel('Reward', color='blue')
    ax.set_title(f'Masks = {masks}', fontsize=11)
    ax.set_xticks(ALPHAS)
    ax.grid(True, alpha=0.3)

    lines = l1 + l2 + l3
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, fontsize=7, loc='best')

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '3_alpha_effect.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved 3_alpha_effect.png")

# ============================================================
# FIGURE 4: Effect of Masks (holding alpha fixed)
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
fig.suptitle('Effect of Mask Count — Holding α Fixed', fontsize=14, fontweight='bold')

for idx, alpha in enumerate(ALPHAS):
    ax = axes[idx]
    masks_vals = []
    best_accs = []
    final_advs = []
    final_sems = []
    for masks in MASKS:
        if (alpha, masks) in data:
            d = data[(alpha, masks)]
            masks_vals.append(masks)
            best_accs.append(d['best_acc'] if d['best_acc'] else d['val_accs'].min())
            final_advs.append(d['advs'][-1] if len(d['advs']) > 0 else 0)
            final_sems.append(d['sems'][-1] if len(d['sems']) > 0 else 0)

    ax2 = ax.twinx()
    l1 = ax.plot(masks_vals, best_accs, 'o-', color='red', linewidth=2, markersize=8, label='Best Val Acc (↓)')
    l2 = ax2.plot(masks_vals, final_advs, 's--', color='blue', linewidth=2, markersize=8, label='Final Adv Rwd (↑)')
    l3 = ax2.plot(masks_vals, final_sems, '^:', color='green', linewidth=2, markersize=8, label='Final Sem Rwd (↑)')

    ax.set_xlabel('Mask Count')
    ax.set_ylabel('Val Accuracy', color='red')
    ax2.set_ylabel('Reward', color='blue')
    ax.set_title(f'α = {alpha}', fontsize=11)
    ax.set_xticks(MASKS)
    ax.grid(True, alpha=0.3)

    lines = l1 + l2 + l3
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, fontsize=7, loc='best')

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '4_masks_effect.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved 4_masks_effect.png")

# ============================================================
# FIGURE 5: Convergence speed comparison
# ============================================================
fig, ax = plt.subplots(1, 1, figsize=(12, 6))
fig.suptitle('Convergence: Steps to Best Val Accuracy', fontsize=14, fontweight='bold')

configs = []
best_steps = []
best_accs_list = []
colors_list = []

for alpha in ALPHAS:
    for masks in MASKS:
        if (alpha, masks) in data:
            d = data[(alpha, masks)]
            configs.append(f'α={alpha}\nm={masks}')
            best_steps.append(d['best_step'] if d['best_step'] else 0)
            best_accs_list.append(d['best_acc'] if d['best_acc'] else 1.0)
            colors_list.append(COLORS[masks])

x_pos = range(len(configs))
bars = ax.bar(x_pos, best_steps, color=colors_list, alpha=0.7, edgecolor='black')

# Annotate with best accuracy
for i, (bar, acc) in enumerate(zip(bars, best_accs_list)):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 50,
            f'acc={acc:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_xticks(x_pos)
ax.set_xticklabels(configs, fontsize=9)
ax.set_ylabel('Steps to Best Checkpoint')
ax.set_xlabel('Configuration')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '5_convergence.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved 5_convergence.png")

# ============================================================
# Print summary table
# ============================================================
print("\n" + "="*100)
print(f"{'Alpha':>6} {'Masks':>6} {'Best Acc':>9} {'Best Step':>10} {'Final Adv':>10} {'Final Sem':>10} {'Final Loss':>11} {'Status':>10}")
print("-"*100)
for alpha in ALPHAS:
    for masks in MASKS:
        if (alpha, masks) in data:
            d = data[(alpha, masks)]
            status = "Done" if len(d['steps']) >= 99 else f"Running ({len(d['steps'])}/99)"
            best_acc = d['best_acc'] if d['best_acc'] else d['val_accs'].min()
            final_adv = d['advs'][-1] if len(d['advs']) > 0 else 0
            final_sem = d['sems'][-1] if len(d['sems']) > 0 else 0
            final_loss = d['losses'][-1] if len(d['losses']) > 0 else 0
            best_step = d['best_step'] if d['best_step'] else 0
            print(f"{alpha:>6.1f} {masks:>6d} {best_acc:>9.4f} {best_step:>10d} {final_adv:>10.4f} {final_sem:>10.4f} {final_loss:>11.4f} {status:>10}")
print("="*100)
