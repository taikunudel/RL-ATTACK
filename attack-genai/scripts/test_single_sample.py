#!/usr/bin/env python3
"""
Test script to run evaluation on a single AdvBench sample.
Verifies the batch moderation optimization works correctly.
"""
import subprocess
import sys

# Run the evaluation script with a single sample (start_idx=0, stop after 1)
cmd = [
    "python", "evaluation_attacker_genai_llama_itself.py",
    "--atker_path", "google-bert/bert-base-uncased",
    "--atker_mode", "untrained",
    "--target_path", "Qwen/Qwen3-8B",
    "--data_name", "harmful_behaviors",
    "--save_to_path", "./test_model.pt",
    "--len_doc_max", "128",
    "--num_doc_masks", "3",  # Only 3 positions to speed up
    "--samples_per_tok", "5",  # Only 5 candidates per position
    "--atk_json_log", "./test_single_sample_output.json",
    "--max_gen_tokens", "32",
    "--server_url", "http://localhost:8001/v1",
    "--use_moderation_api",
    "--start_idx", "0",
]

print("Running command:")
print(" ".join(cmd))
print("\n" + "="*60 + "\n")

# Run and stream output
process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

for line in process.stdout:
    print(line, end='')

process.wait()
print("\n" + "="*60)
print(f"Exit code: {process.returncode}")
print("Output saved to: test_single_sample_output.json")
