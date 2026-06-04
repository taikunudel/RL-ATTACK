"""
Test AdvBench samples 11-20 with both OpenAI and Gemini Moderation APIs.
Compare structure and flagged results.
"""
from openai import OpenAI
from google import genai
from datasets import load_dataset
from pydantic import BaseModel
from typing import List
import os
import json

# Load API keys from .env
if os.path.exists(".env"):
    with open(".env", "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("OPENAI_API_KEY="):
                os.environ["OPENAI_API_KEY"] = line.split("=", 1)[1]
            if line.startswith("GOOGLE_API_KEY="):
                os.environ["GOOGLE_API_KEY"] = line.split("=", 1)[1]

# Load AdvBench samples 11-20
print("Loading AdvBench samples 11-20...")
advbench = load_dataset("walledai/AdvBench")
samples_11_20 = advbench['train']['prompt'][10:20]

print("\nSamples 11-20:")
for i, s in enumerate(samples_11_20):
    print(f"[{i+11}] {s[:70]}...")

# ============== RUN OPENAI ==============
print("\n\n========== OpenAI Moderation API ==========")
openai_client = OpenAI()
openai_response = openai_client.moderations.create(
    model="omni-moderation-latest",
    input=samples_11_20,
)
openai_output = openai_response.model_dump()

with open("advbench_11_20_openai.json", "w") as f:
    json.dump(openai_output, f, indent=2)
print("Saved to advbench_11_20_openai.json")

# ============== RUN GEMINI (using the moderation mimic) ==============
print("\n========== Gemini Moderation Mimic ==========")

# Import from mimic script
from mimic_moderation_with_gemini import FEWSHOT_PROMPT, build_prompt, transform_to_openai_format, ModerationResponse

gemini_client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
prompt_text = build_prompt(samples_11_20)

print("Calling Gemini API...")
gemini_response = gemini_client.models.generate_content(
    model="gemini-2.0-flash",
    contents=prompt_text,
    config={
        "response_mime_type": "application/json",
        "response_json_schema": ModerationResponse.model_json_schema(),
    },
)

gemini_raw = json.loads(gemini_response.text)
gemini_output = transform_to_openai_format(gemini_raw)

with open("advbench_11_20_gemini.json", "w") as f:
    json.dump(gemini_output, f, indent=2)
print("Saved to advbench_11_20_gemini.json")

# ============== COMPARE STRUCTURE ==============
print("\n\n========== STRUCTURE COMPARISON ==========")

# Top-level keys
openai_keys = set(openai_output.keys())
gemini_keys = set(gemini_output.keys())
print(f"Top-level keys match: {openai_keys == gemini_keys} ({openai_keys})")

# Result keys
openai_result_keys = set(openai_output["results"][0].keys())
gemini_result_keys = set(gemini_output["results"][0].keys())
print(f"Result keys match: {openai_result_keys == gemini_result_keys}")
if openai_result_keys != gemini_result_keys:
    print(f"  OpenAI: {sorted(openai_result_keys)}")
    print(f"  Gemini: {sorted(gemini_result_keys)}")

# Category keys
openai_cat_keys = set(openai_output["results"][0]["categories"].keys())
gemini_cat_keys = set(gemini_output["results"][0]["categories"].keys())
print(f"Category keys match: {openai_cat_keys == gemini_cat_keys} ({len(openai_cat_keys)} keys)")

# ============== COMPARE FLAGGED RESULTS ==============
print("\n\n========== FLAGGED RESULTS COMPARISON ==========")
print(f"{'#':<3} {'OpenAI':<8} {'Gemini':<8} {'Match':<6} Sample")
print("-" * 80)

matches = 0
for i, (sample, openai_result, gemini_result) in enumerate(zip(samples_11_20, openai_output["results"], gemini_output["results"])):
    openai_flagged = openai_result["flagged"]
    gemini_flagged = gemini_result["flagged"]
    match = "✅" if openai_flagged == gemini_flagged else "❌"
    if openai_flagged == gemini_flagged:
        matches += 1
    print(f"{i+11:<3} {str(openai_flagged):<8} {str(gemini_flagged):<8} {match:<6} {sample[:40]}...")

print(f"\n\nAgreement: {matches}/10 ({matches*10}%)")
