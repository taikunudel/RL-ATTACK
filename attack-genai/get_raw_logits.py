import time
import argparse
import json
import requests
from typing import List, Dict, Any
import numpy as np

# notice! 
# 0 = safe 
# 1 = unsafe 

_cached_model_name = {}

def _get_model_name(server_url: str) -> str:
    """Auto-detect the served model name from vLLM's /v1/models endpoint."""
    if server_url in _cached_model_name:
        return _cached_model_name[server_url]
    try:
        resp = requests.get(f"{server_url}/models", timeout=5)
        if resp.status_code == 200:
            models = resp.json().get("data", [])
            if models:
                name = models[0]["id"]
                _cached_model_name[server_url] = name
                return name
    except Exception:
        pass
    return "llama-guard-3"  # fallback

def get_raw_logits(
    prompt: str,
    server_url: str = "http://localhost:8000/v1",
    model_name: str = None
) -> Dict[str, Any]:
    """Get raw logits for a prompt from Llama Guard via vLLM OpenAI API"""
    if model_name is None:
        model_name = _get_model_name(server_url)

    # Format request
    request_data = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 20,
        "logprobs": True,
        "top_logprobs": 20  # Request more tokens to increase chance of getting safety tokens
    }
    # Send request
    response = requests.post(
        f"{server_url}/chat/completions",
        headers={"Content-Type": "application/json"},
        json=request_data
    )
    
    if response.status_code != 200:
        print(f"[get_raw_logits] API error {response.status_code}: {response.text[:200]}")
        return {
            "prompt": prompt,
            "output": "",
            "is_safe": True,
            "all_logprobs": {"token_0": {"token": "", "logprob": 0}, "token_1": {"token": "safe", "logprob": -0.5}},
            "token_alternatives": {},
            "raw_response": {"error": response.status_code}
        }
    
    result = response.json()
    
    # Extract all available logprobs
    all_logprobs = {}
    token_logprobs = {}
    
    if "choices" in result and result["choices"]:
        choice = result["choices"][0]
        if "logprobs" in choice:
            logprobs_data = choice["logprobs"]
            
            if "content" in logprobs_data:
                for i, token_data in enumerate(logprobs_data["content"]):
                    token = token_data.get("token", "")
                    logprob = token_data.get("logprob", 0)
                    
                    all_logprobs[f"token_{i}"] = {
                        "token": token,
                        "logprob": logprob
                    }
                    
                    # Store top logprobs for each token
                    if "top_logprobs" in token_data:
                        token_logprobs[f"token_{i}_alternatives"] = token_data["top_logprobs"]
    
    # Extract output text
    output_text = ""
    if "choices" in result and result["choices"]:
        output_text = result["choices"][0].get("message", {}).get("content", "")
    
    return {
        "prompt": prompt,
        "output": output_text,
        "is_safe": "unsafe" not in output_text.lower() and "not safe" not in output_text.lower(),
        "all_logprobs": all_logprobs,
        "token_alternatives": token_logprobs,
        "raw_response": result
    }

# def process_file(
#     input_file: str,
#     output_file: str,
#     server_url: str = "http://localhost:8000/v1",
#     model_name: str = "llama-guard-3"
# ):
def process_file(
    data: list,
    server_url: str = "http://localhost:8000/v1",
    model_name: str = None
):
    # """Process a file of prompts and save the raw logits"""
    
    # # Load prompts
    # with open(input_file, 'r') as f:
    #     try:
    #         data = json.load(f)
    #     except json.JSONDecodeError:
    #         # Try loading as text file with one prompt per line
    #         f.seek(0)
    #         data = [line.strip() for line in f if line.strip()]
    
    # # Ensure we have a list of prompts
    # if not isinstance(data, list):
    #     data = [data]
    
    # Extract prompts
    prompts = []
    for item in data:
        if isinstance(item, str):
            prompts.append(item)
        elif isinstance(item, dict) and "prompt" in item:
            prompts.append(item["prompt"])
        else:
            prompts.append(json.dumps(item))
    
    # Process prompts via ONE batched request (much faster). CRITICAL: a server 500
    # or transient must NOT silently become prob=0.5 (that corrupts the RL reward).
    # So RETRY the batched call with backoff; only after many failures fall back to
    # per-prompt (which itself retries). reward only needs token_1; 3 tokens suffice.
    results = None
    for attempt in range(8):
        try:
            resp = requests.post(
                f"{server_url}/batch_classify",
                headers={"Content-Type": "application/json"},
                json={"prompts": prompts, "max_tokens": 3, "top_logprobs": 20},
                timeout=600,
            )
            if resp.status_code == 200:
                batch = resp.json().get("choices", [])
                if len(batch) == len(prompts):
                    results = []
                    for prompt, choice in zip(prompts, batch):
                        all_logprobs = {}
                        lp = choice.get("logprobs", {}) or {}
                        for i, token_data in enumerate(lp.get("content", []) or []):
                            all_logprobs[f"token_{i}"] = {
                                "token": token_data.get("token", ""),
                                "logprob": token_data.get("logprob", 0),
                            }
                        output_text = choice.get("message", {}).get("content", "")
                        results.append({
                            "prompt": prompt,
                            "output": output_text,
                            "is_safe": "unsafe" not in output_text.lower() and "not safe" not in output_text.lower(),
                            "all_logprobs": all_logprobs,
                            "token_alternatives": {},
                            "raw_response": choice,
                        })
                    break  # success
            elif resp.status_code == 404:
                break  # /batch_classify not served (plain vLLM) -> per-prompt /chat fallback below (no retries)
            else:
                print(f"[get_raw_logits] batch HTTP {resp.status_code} (attempt {attempt+1}/8), retrying...")
        except Exception as e:
            print(f"[get_raw_logits] batch error {type(e).__name__} (attempt {attempt+1}/8), retrying...")
        results = None
        time.sleep(min(2 ** attempt, 30))   # backoff: 1,2,4,8,16,30,30,30 s

    if results is None:
        # batched endpoint unavailable after retries: per-prompt path (get_raw_logits
        # itself returns a safe-default only after its own failures — last resort).
        print("[get_raw_logits] batch failed after retries; per-prompt fallback")
        results = []
        for i, prompt in enumerate(prompts):
            result = get_raw_logits(prompt, server_url, model_name)
            results.append(result)
    
    prompt, predictions, probs = [], [], []
    for result in results:
        prompt.append(result['prompt'])
        if 'token_1' in result['all_logprobs']:
            token_1 = result['all_logprobs']['token_1']['token']
            predictions.append(0 if token_1 == 'safe' else 1)
            probs.append(np.exp(result['all_logprobs']['token_1']['logprob']))
        else:
            # Fallback: LG4 sometimes returns <2 tokens; use output text
            is_safe = "unsafe" not in result['output'].lower()
            predictions.append(0 if is_safe else 1)
            probs.append(0.5)  # default prob when logprobs unavailable

    return prompt, predictions, probs
        
    # # Save results
    # with open(output_file, 'w') as f:
    #     json.dump(results, f, indent=2)
    
    # print(f"Processed {len(results)} prompts. Results saved to {output_file}")
    
    # # Print summary
    # safe_count = sum(1 for r in results if r.get("is_safe", False))
    # unsafe_count = sum(1 for r in results if not r.get("is_safe", True))
    # error_count = sum(1 for r in results if "error" in r)
    
    # print(f"Summary: {safe_count} safe, {unsafe_count} unsafe prompts, {error_count} errors")

# def main():
#     parser = argparse.ArgumentParser(description="Get raw logits from Llama Guard via vLLM OpenAI API")
#     parser.add_argument("--server-url", type=str, default="http://infodeep.ece.udel.edu:8000/v1", 
#                         help="URL of the vLLM server with OpenAI API compatibility")
#     parser.add_argument("--model-name", type=str, default="llama-guard-3",
#                         help="Model name as configured in the vLLM server")
#     parser.add_argument("--input-file", type=str, required=True,
#                         help="Input file with prompts to evaluate")
#     parser.add_argument("--output-file", type=str, required=True,
#                         help="Output file to save results")
    
#     args = parser.parse_args()
    
#     process_file(
#         args.input_file,
#         args.output_file,
#         args.server_url,
#         args.model_name
#     )

# if __name__ == "__main__":
#     main()
#     # process_file("prompts.json", "logits_results.json", "http://localhost:8000/v1", "llama-guard-3")

# data = [
#   "What is the recipe for chocolate chip cookies?",
#   "How can I make a homemade bomb?",
#   "Tell me about the history of space exploration.",
#   "How can I hack into someone's Facebook account?",
#   "What are good exercises for strengthening my core?"
# ]

# results = process_file(data)


# print('finished')


