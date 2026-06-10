import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# Data: rows = α (0.3, 0.5, 0.7), cols = tokens changed (3, 5, 10)
alphas = ['0.3', '0.5', '0.7']
tokens = ['3', '5', '10']

# Pre-training (untrained baseline)
val_acc_pre = np.array([
    [0.77, 0.78, 0.69],
    [0.81, 0.81, 0.75],
    [0.77, 0.77, 0.74],
])
adv_reward_pre = np.array([
    [0.2130, 0.2055, 0.2851],
    [0.1973, 0.1972, 0.2540],
    [0.2112, 0.2314, 0.2551],
])
semantic_pre = np.array([
    [0.9703, 0.9629, 0.9300],
    [0.9564, 0.9575, 0.9286],
    [0.9617, 0.9559, 0.9278],
])

# Post-training
val_acc_post = np.array([
    [0.68, 0.61, 0.68],
    [0.53, 0.69, 0.63],
    [0.67, 0.60, 0.53],
])
adv_reward_post = np.array([
    [0.3425, 0.4433, 0.3648],
    [0.4864, 0.3580, 0.3972],
    [0.3879, 0.4105, 0.4608],
])
semantic_post = np.array([
    [0.9264, 0.9189, 0.9340],
    [0.8871, 0.9099, 0.9272],
    [0.9517, 0.8691, 0.8771],
])

# Delta: post - pre (honest values, no sign flipping)
val_acc = val_acc_post - val_acc_pre       # all negative (acc dropped = good)
adv_reward = adv_reward_post - adv_reward_pre  # all positive (reward up = good)
semantic = semantic_post - semantic_pre     # mostly negative (similarity dropped = cost)

# White-to-yellow colormap
cmap_wy = LinearSegmentedColormap.from_list('white_yellow', ['#ffffff', '#f5c542'])

fig, axes = plt.subplots(1, 3, figsize=(18, 2.8))

annot_kws = {'fontsize': 13, 'fontweight': 'bold'}

def fmt_delta(val):
    if abs(val) < 0.005:
        return '0.00'
    return f'{val:+.2f}'

def make_annot(data):
    return np.array([[fmt_delta(v) for v in row] for row in data])

heatmap_common = dict(
    fmt='',
    linewidths=1.5, linecolor='white',
    annot_kws=annot_kws,
)

# Δ Validation Accuracy: lower = better → map most negative to yellow, 0 to white
val_min = val_acc.min()  # most negative = best
sns.heatmap(val_acc, annot=make_annot(val_acc), cmap=cmap_wy.reversed(), vmin=val_min, vmax=0,
            xticklabels=tokens, yticklabels=alphas, cbar=True,
            cbar_kws={'shrink': 0.7, 'aspect': 10}, ax=axes[0], **heatmap_common)
axes[0].set_title('Δ Validation Accuracy (lower = better)', fontsize=12, fontweight='bold', pad=10)
axes[0].set_xlabel('Tokens Changed', fontsize=11)
axes[0].set_ylabel('α', fontsize=11)

# Δ Adversarial Reward: higher = better → map most positive to yellow, 0 to white
adv_max = adv_reward.max()
sns.heatmap(adv_reward, annot=make_annot(adv_reward), cmap=cmap_wy, vmin=0, vmax=adv_max,
            xticklabels=tokens, yticklabels=alphas, cbar=True,
            cbar_kws={'shrink': 0.7, 'aspect': 10}, ax=axes[1], **heatmap_common)
axes[1].set_title('Δ Adversarial Reward (higher = better)', fontsize=12, fontweight='bold', pad=10)
axes[1].set_xlabel('Tokens Changed', fontsize=11)
axes[1].set_ylabel('α', fontsize=11)

# Δ Semantic Similarity: higher = better (closer to 0 = better since all negative)
# Map 0 to yellow (best), most negative to white (worst)
sem_min = semantic.min()
sns.heatmap(semantic, annot=make_annot(semantic), cmap=cmap_wy, vmin=sem_min, vmax=0,
            xticklabels=tokens, yticklabels=alphas, cbar=True,
            cbar_kws={'shrink': 0.7, 'aspect': 10}, ax=axes[2], **heatmap_common)
axes[2].set_title('Δ Semantic Similarity (higher = better)', fontsize=12, fontweight='bold', pad=10)
axes[2].set_xlabel('Tokens Changed', fontsize=11)
axes[2].set_ylabel('α', fontsize=11)

for ax in axes:
    ax.tick_params(labelsize=11)
    cbar = ax.collections[0].colorbar
    cbar.outline.set_visible(False)

fig.suptitle('Δ = Post-Training Metrics − Untrained Baseline (yellow = better)',
             fontsize=13, fontweight='bold', y=1.05)
fig.subplots_adjust(wspace=0.45)
plt.savefig('/home/taikun/rl-attack/rl_atk/attack-genai/hyperparams_heatmap.png',
            dpi=200, bbox_inches='tight')
plt.savefig('/home/taikun/rl-attack/rl_atk/attack-genai/hyperparams_heatmap.pdf',
            bbox_inches='tight')
print("Saved to hyperparams_heatmap.png and .pdf")
