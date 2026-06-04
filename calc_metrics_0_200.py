import json
import numpy as np

filepath = "/usa/taikun/rl-attack/rl_atk/attack-genai/eva_results/eva_0119_220946_bert-base-uncased_qwen3-8b_doc_trained_harmul_strings_5_150.json"

try:
    with open(filepath, 'r') as f:
        data = json.load(f)
except Exception as e:
    print(f"Error loading file: {e}")
    exit(1)

# Slice 0-200
subset = data[0:200]
print(f"Total samples in 0-200 range: {len(subset)}")

# Filter out 'illicit'
# Checking 'moderation_info' for 'illicit'
filtered_data = []
removed_count = 0
for item in subset:
    mod_info = item.get('moderation_info', '')
    if 'illicit' in mod_info.lower():
        removed_count += 1
    else:
        filtered_data.append(item)

print(f"Removed 'illicit' count: {removed_count}")
print(f"Remaining samples: {len(filtered_data)}")

if len(filtered_data) == 0:
    print("No samples remaining.")
    exit(0)

# Calculate metrics for remaining
queries = [item.get('queries_used', 0) for item in filtered_data]
perturbations = [item.get('perturbation_rate', 0) for item in filtered_data]
uses = [item.get('USEs', 0) for item in filtered_data]

# Assuming adv_pred_label == 0 means Attack Success (Jailbreak) and 1 means Failure (Refusal)
# Or we look at true_label.
# Let's count labels.
adv_labels = [item.get('adv_pred_label') for item in filtered_data]
successes = [1 for l in adv_labels if l == 0] # Assuming 0 is success
failures = [1 for l in adv_labels if l == 1]

# Check assumption by printing a sample text for label 0 and 1
print("\n--- Label Inspection ---")
for label in [0, 1]:
    found = False
    for item in filtered_data:
        if item.get('adv_pred_label') == label:
            print(f"Label {label} snippet: {item.get('adv_response_only')[:100]}...")
            found = True
            break
    if not found:
        print(f"No samples found for Label {label}")

asr = len(successes) / len(filtered_data) * 100
avg_queries = np.mean(queries)
avg_perturbation = np.mean(perturbations)
avg_use = np.mean(uses)


src_labels = [item.get('src_pred_label') for item in filtered_data]
# Assuming 1 is Safe (Success for Defender)
src_robustness = len([1 for l in src_labels if l == 1]) / len(filtered_data) * 100

print("\n--- Metrics (Excluding Illicit) ---")
print(f"Attack Success Rate (ASR): {asr:.2f}%")
print(f"Original Model Robustness (Safe responses): {src_robustness:.2f}%")
print(f"Avg Queries: {avg_queries:.2f}")
print(f"Avg Perturbation Rate: {avg_perturbation:.4f}")
print(f"Avg USE Score: {avg_use:.4f}")
