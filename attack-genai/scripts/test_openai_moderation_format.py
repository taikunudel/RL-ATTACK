from openai import OpenAI
import os
import json

# Load API key from .env if not in environment
if "OPENAI_API_KEY" not in os.environ:
    try:
        if os.path.exists(".env"):
            with open(".env", "r") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("OPENAI_API_KEY="):
                        os.environ["OPENAI_API_KEY"] = line.split("=", 1)[1]
                        break
    except Exception as e:
        print(f"Warning: Could not load .env: {e}")

if "OPENAI_API_KEY" not in os.environ:
    print("Error: OPENAI_API_KEY not found in environment or .env file.")
    exit(1)

client = OpenAI()

test_inputs = [
    "I love puppies and sunshine.",
    "I want to make a bomb to kill people."
]

print(f"Testing with {len(test_inputs)} inputs...")

try:
    response = client.moderations.create(
        model="omni-moderation-latest",
        input=test_inputs,
    )

    print("\n--- Raw Response Object (JSON) ---")
    # This shows the exact structure needed to mimic
    json_output = response.model_dump_json(indent=2)
    print(json_output)

    # Save input and output to files
    with open("moderation_input.json", "w") as f:
        json.dump(test_inputs, f, indent=2)
    print("\nSaved input to moderation_input.json")

    with open("moderation_output.json", "w") as f:
        f.write(json_output)
    print("Saved output to moderation_output.json")
    
except Exception as e:
    print(f"Error calling API: {e}")
