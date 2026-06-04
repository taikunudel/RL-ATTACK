"""
Step 1: Load first 10 AdvBench prompts and run OpenAI Moderation API on them.
Save results for use as few-shot examples.
"""
from openai import OpenAI
from datasets import load_dataset
import os
import json

# Load API key from .env
if "OPENAI_API_KEY" not in os.environ:
    if os.path.exists(".env"):
        with open(".env", "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("OPENAI_API_KEY="):
                    os.environ["OPENAI_API_KEY"] = line.split("=", 1)[1]
                    break

# Load AdvBench dataset
print("Loading AdvBench dataset...")
advbench = load_dataset("walledai/AdvBench")
first_10_prompts = advbench['train']['prompt'][:10]

print(f"\nFirst 10 AdvBench prompts:")
for i, p in enumerate(first_10_prompts):
    print(f"[{i+1}] {p[:80]}...")

# Run OpenAI Moderation API
print("\n\nRunning OpenAI Moderation API on first 10 AdvBench prompts...")
client = OpenAI()

response = client.moderations.create(
    model="omni-moderation-latest",
    input=first_10_prompts,
)

output = response.model_dump()

# Save full output
with open("advbench_10_openai_moderation.json", "w") as f:
    json.dump(output, f, indent=2)
print("Saved to advbench_10_openai_moderation.json")

# Create few-shot examples in a format suitable for prompting
few_shot_examples = []
for i, (prompt, result) in enumerate(zip(first_10_prompts, output["results"])):
    # Simplify the result for few-shot (only core categories, exclude slash variants)
    core_cats = ["harassment", "harassment_threatening", "hate", "hate_threatening",
                 "illicit", "illicit_violent", "self_harm", "self_harm_instructions",
                 "self_harm_intent", "sexual", "sexual_minors", "violence", "violence_graphic"]
    
    categories = {k: result["categories"].get(k, False) for k in core_cats}
    category_scores = {k: round(result["category_scores"].get(k, 0.0), 4) for k in core_cats}
    
    example = {
        "input": prompt,
        "output": {
            "flagged": result["flagged"],
            "categories": categories,
            "category_scores": category_scores
        }
    }
    few_shot_examples.append(example)

# Save few-shot examples
with open("advbench_10_fewshot.json", "w") as f:
    json.dump(few_shot_examples, f, indent=2)
print("Saved few-shot examples to advbench_10_fewshot.json")

# Print summary
print("\n\n=== SUMMARY ===")
for i, ex in enumerate(few_shot_examples):
    flagged = ex["output"]["flagged"]
    flagged_cats = [k for k, v in ex["output"]["categories"].items() if v]
    print(f"[{i+1}] Flagged: {flagged}")
    if flagged:
        print(f"    Categories: {', '.join(flagged_cats)}")
