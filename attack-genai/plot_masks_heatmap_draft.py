import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# Axes: masks (rows) × candidates (cols) — but we only have cands=20
# So we simulate lower candidate budgets from the masks=3 data (post-hoc)
# and mark masks=10 as projected

masks_labels = ['3', '5', '10']

# Real data
# masks=3, cands=20: atk_acc=41.00%, pert=4.41%, USE=0.9743
# masks=5, cands=20: atk_acc=18.65%, pert=5.88%, USE=0.9714 (65/100, partial)
# masks=10, cands=20: projected

atk_acc = np.array([41.00, 18.65, 8.0])   # masks=10 projected
ori_acc = np.array([81.00, 67.88, 55.0])   # masks=10 projected
pert    = np.array([4.41,  5.88,  9.5])    # masks=10 projected
use     = np.array([0.9743, 0.9714, 0.962]) # masks=10 projected

is_projected = [False, False, True]  # masks=10 is imaginary
is_partial   = [False, True, False]  # masks=5 is 65/100

fig, axes = plt.subplots(1, 4, figsize=(20, 3.2))

cmap_good_low = LinearSegmentedColormap.from_list('wg', ['#ffffff', '#2ecc71'])  # lower=greener=better
cmap_good_high = LinearSegmentedColormap.from_list('wy', ['#ffffff', '#f5c542'])  # higher=yellower=better
cmap_bad_high = LinearSegmentedColormap.from_list('wr', ['#ffffff', '#e74c3c'])   # higher=redder=worse

panels = [
    # (title, data, unit, cmap, vmin, vmax)
    # For atk_acc: 0%=perfect attack (green), 100%=no attack (white)
    ('Attack Accuracy ↓', atk_acc, '%', cmap_good_low.reversed(), 0, 100),
    ('Original Accuracy', ori_acc, '%', cmap_good_low.reversed(), 0, 100),
    # For pert: 0%=no perturbation (white), 15%=high perturbation (red)
    ('Perturbation Rate', pert,    '%', cmap_bad_high, 0, 15),
    # For USE: 0.9=low (white), 1.0=perfect (yellow)
    ('USE Similarity',    use,     '',  cmap_good_high, 0.90, 1.0),
]

for ax, (title, data, unit, cmap, vmin, vmax) in zip(axes, panels):
    # Reshape to 1×3 for heatmap
    hm_data = data.reshape(1, -1)

    # Build annotation strings with markers
    annot = []
    for i, v in enumerate(data):
        s = f'{v:.2f}{unit}'
        if is_projected[i]:
            s += '\n(projected)'
        elif is_partial[i]:
            s += '\n(65/100)'
        annot.append(s)
    annot = np.array([annot])

    sns.heatmap(hm_data, annot=annot, fmt='', cmap=cmap,
                vmin=vmin, vmax=vmax,
                xticklabels=masks_labels, yticklabels=['cands=20'],
                linewidths=2, linecolor='white',
                annot_kws={'fontsize': 12, 'fontweight': 'bold'},
                cbar=False, ax=ax)

    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel('Masks (tokens changed)', fontsize=11)
    ax.tick_params(labelsize=11)

    # Hatching for projected cell
    for i, proj in enumerate(is_projected):
        if proj:
            ax.add_patch(plt.Rectangle((i, 0), 1, 1, fill=False,
                         hatch='///', edgecolor='gray', linewidth=0))

fig.suptitle('Evaluation Heatmap: Trained Attacker vs GPT-3.5-turbo (OpenAI Moderation Judge)',
             fontsize=13, fontweight='bold', y=1.08)
fig.subplots_adjust(wspace=0.35)

out_path = '/home/taikun/rl-attack/rl_atk/attack-genai/masks_heatmap_draft.png'
plt.savefig(out_path, dpi=200, bbox_inches='tight')
print(f"Saved to {out_path}")
