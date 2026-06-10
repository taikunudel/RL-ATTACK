#!/usr/bin/env python3
"""
Test batch call to Moderation API - randomly sample 50 from AdvBench
"""

import os
from dotenv import load_dotenv
load_dotenv()  # OPENAI_API_KEY comes from .env

from openai import OpenAI
from datasets import load_dataset
import json
import random

client = OpenAI()

# Load AdvBench dataset
print("Loading AdvBench dataset...")
advbench = load_dataset("walledai/AdvBench")
all_prompts = advbench['train']['prompt']

# Randomly sample 50
random.seed()  # Use current time as seed for randomness
test_texts = random.sample(all_prompts, 50)

print(f"Randomly sampled {len(test_texts)} texts from AdvBench")
print("=" * 60)
print("First 3 samples:")
for i, t in enumerate(test_texts[:3]):
    print(f"  [{i}] {t[:60]}...")
print("=" * 60)

# Send ALL 50 in ONE call (batch mode)
print(f"Sending {len(test_texts)} samples in ONE API call...")
response = client.moderations.create(
    model="omni-moderation-latest",
    input=test_texts
)

print(f"Input length: {len(test_texts)}, Output length: {len(response.results)}")

# Save raw JSON to file
raw_data = response.model_dump()
raw_data['input_texts'] = test_texts  # Also save the input texts
output_file = "/usa/taikun/rl-attack/rl_atk/attack-genai/test_moderation_raw_output.json"

with open(output_file, 'w') as f:
    json.dump(raw_data, f, indent=2)

print(f"Raw JSON saved to: {output_file}")

# Summary
flagged_count = sum(1 for r in response.results if r.flagged)
print(f"\nSummary: {flagged_count} flagged, {len(response.results) - flagged_count} not flagged")
