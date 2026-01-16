#!/usr/bin/env python3
"""
Compare outputs from Qwen3-8B and Llama-3-8B models.
Both models are hosted via vLLM.
"""

from openai import OpenAI

# Model configurations
MODELS = {
    "qwen3-8b": {
        "base_url": "http://localhost:8002/v1",
        "model_name": "qwen3-8b"
    },
    "llama-3-8b": {
        "base_url": "http://localhost:8004/v1",
        "model_name": "llama-3-8b"
    }
}

def query_model(model_key: str, prompt: str, max_tokens: int = 256, temperature: float = 0.7) -> str:
    """Query a model and return its response."""
    config = MODELS[model_key]
    client = OpenAI(
        base_url=config["base_url"],
        api_key="EMPTY"  # vLLM doesn't require an API key
    )
    
    response = client.chat.completions.create(
        model=config["model_name"],
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_tokens,
        temperature=temperature
    )
    
    return response.choices[0].message.content

def compare_outputs(prompt: str, max_tokens: int = 256, temperature: float = 0.7):
    """Compare outputs from both models for the same prompt."""
    print("=" * 80)
    print(f"PROMPT: {prompt}")
    print("=" * 80)
    
    for model_key in MODELS:
        print(f"\n{'─' * 40}")
        print(f"MODEL: {model_key.upper()}")
        print(f"{'─' * 40}")
        try:
            response = query_model(model_key, prompt, max_tokens, temperature)
            print(response)
        except Exception as e:
            print(f"ERROR: {e}")
    
    print("\n" + "=" * 80 + "\n")

def interactive_mode():
    """Interactive mode to chat with both models."""
    print("\n" + "=" * 80)
    print("INTERACTIVE MODEL COMPARISON")
    print("Type your prompts to compare Qwen3-8B and Llama-3-8B outputs")
    print("Type 'quit' or 'exit' to stop")
    print("=" * 80 + "\n")
    
    while True:
        try:
            prompt = input("\n[Your prompt] > ").strip()
            if prompt.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            if not prompt:
                continue
            compare_outputs(prompt)
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break

if __name__ == "__main__":
    # Test prompts for comparison
    test_prompts = [
        "What is 2+2?",
        "Write a haiku about coding.",
        "Explain what machine learning is in one sentence."
    ]
    
    print("\n" + "=" * 80)
    print("RUNNING COMPARISON TESTS")
    print("=" * 80)
    
    for prompt in test_prompts:
        compare_outputs(prompt, max_tokens=128, temperature=0.0)  # Use temp=0 for reproducibility
    
    # Enter interactive mode
    interactive_mode()
