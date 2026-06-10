import json
import numpy as np
import matplotlib.pyplot as plt

with open('/home/taikun/rl-attack/rl_atk/attack-genai/eva_results/'
          'eva_0324_bert-base-uncased_gpt-3.5-turbo_doc_trained_jbb_behaviors_3_20_openaimod.json') as f:
    data = json.load(f)

# Extract per-sample data
queries = np.array([d['queries_used'] for d in data])
adv_labels = np.array([d['adv_pred_label'] for d in data])
true_labels = np.array([d['true_label'] for d in data])
judge_correct = (adv_labels == true_labels).astype(float)
uses = np.array([d['USEs'] for d in data])
perts = np.array([d['perturbation_rate'] for d in data])
n = len(data)

# Simulate different candidate counts by capping queries
# Attack loop: for each of 3 positions, try C candidates sequentially
# So max queries = 3 * C
candidates_list = [5, 10, 15, 20]
caps = [3 * c for c in candidates_list]  # 15, 30, 45, 60

sim_acc = []   # judge accuracy (lower = better attack)
sim_use = []   # USE similarity
sim_pert = []  # perturbation rate

for cap in caps:
    # Samples within budget: use actual results
    # Samples over budget: attack failed → judge correct, USE=1.0, pert=0
    within = queries <= cap
    acc_vals = np.where(within, judge_correct, 1.0)  # failed = judge correct
    use_vals = np.where(within, uses, 1.0)            # failed = no change
    pert_vals = np.where(within, perts, 0.0)           # failed = no perturbation

    sim_acc.append(acc_vals.mean())
    sim_use.append(use_vals.mean())
    sim_pert.append(pert_vals.mean())

# Plot as bar charts
fig, axes = plt.subplots(1, 3, figsize=(15, 3.5))
x = np.arange(len(candidates_list))
bar_width = 0.5
xlabels = [str(c) for c in candidates_list]

# Panel 1: Accuracy After Attack
bars1 = axes[0].bar(x, sim_acc, bar_width, color='#2563eb', alpha=0.8)
axes[0].set_xticks(x)
axes[0].set_xticklabels(xlabels)
axes[0].set_xlabel('Candidates per Token', fontsize=11)
axes[0].set_ylabel('Accuracy After Attack', fontsize=11)
axes[0].set_title('Accuracy After Attack', fontsize=12, fontweight='bold')
axes[0].set_ylim(0, 1.0)
for bar, val in zip(bars1, sim_acc):
    axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f'{val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Panel 2: USE Similarity
bars2 = axes[1].bar(x, sim_use, bar_width, color='#2563eb', alpha=0.8)
axes[1].set_xticks(x)
axes[1].set_xticklabels(xlabels)
axes[1].set_xlabel('Candidates per Token', fontsize=11)
axes[1].set_ylabel('USE Similarity', fontsize=11)
axes[1].set_title('Semantic Similarity (USE)', fontsize=12, fontweight='bold')
axes[1].set_ylim(0.9, 1.01)
for bar, val in zip(bars2, sim_use):
    axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                 f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Panel 3: Perturbation Rate
bars3 = axes[2].bar(x, sim_pert, bar_width, color='#2563eb', alpha=0.8)
axes[2].set_xticks(x)
axes[2].set_xticklabels(xlabels)
axes[2].set_xlabel('Candidates per Token', fontsize=11)
axes[2].set_ylabel('Perturbation Rate', fontsize=11)
axes[2].set_title('Perturbation Rate', fontsize=12, fontweight='bold')
for bar, val in zip(bars3, sim_pert):
    axes[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                 f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

for ax in axes:
    ax.tick_params(labelsize=10)
    ax.grid(alpha=0.2, axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

fig.suptitle('Simulated Candidate Budget (3 Tokens Changed, GPT-3.5-Turbo, OpenAI Moderation Judge)',
             fontsize=12, fontweight='bold', y=1.03)
fig.subplots_adjust(wspace=0.35)
plt.savefig('/home/taikun/rl-attack/rl_atk/attack-genai/queries_vs_metrics.png',
            dpi=200, bbox_inches='tight')
plt.savefig('/home/taikun/rl-attack/rl_atk/attack-genai/queries_vs_metrics.pdf',
            bbox_inches='tight')
print("Saved to queries_vs_metrics.png and .pdf")

# Print summary
print(f"\n{'Candidates':>12} {'Cap':>5} {'Acc':>8} {'USE':>8} {'Pert':>8} {'Succeeded':>10}")
for c, cap, a, u, p in zip(candidates_list, caps, sim_acc, sim_use, sim_pert):
    succeeded = (queries <= cap).sum()
    print(f'{c:>12} {cap:>5} {a:>8.4f} {u:>8.4f} {p:>8.4f} {succeeded:>7}/{n}')
