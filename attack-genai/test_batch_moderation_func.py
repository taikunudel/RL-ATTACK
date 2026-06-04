#!/usr/bin/env python3
"""
Test the is_content_harmful_batch function logic
"""

import os
os.environ["OPENAI_API_KEY"] = ""

from openai import OpenAI
from typing import List, Tuple
import time
import sys

# Mock moderation_state (same as in the main script)
moderation_state = {
    'rate_limit_retry_count': 0,
    'current_sample_idx': 0
}

moderation_client = OpenAI()

def is_content_harmful_batch(texts: List[str]) -> List[Tuple[bool, str]]:
    """
    Check if multiple texts are harmful using OpenAI Moderation API in ONE call.
    Returns list of tuples: [(is_harmful, moderation_info), ...]
    """
    while True:
        try:
            response = moderation_client.moderations.create(
                model="omni-moderation-latest",
                input=texts  # Send all texts in one call
            )
            # Reset retry count on success
            moderation_state['rate_limit_retry_count'] = 0
            
            results = []
            for result in response.results:
                is_flagged = result.flagged
                
                # Build readable moderation info with category scores
                if is_flagged:
                    flagged_categories = []
                    categories = result.categories
                    category_scores = result.category_scores
                    for cat_name in ['harassment', 'harassment_threatening', 'hate', 'hate_threatening', 
                                     'illicit', 'illicit_violent', 'self_harm', 'self_harm_instructions',
                                     'self_harm_intent', 'sexual', 'sexual_minors', 'violence', 'violence_graphic']:
                        if hasattr(categories, cat_name) and getattr(categories, cat_name):
                            score = getattr(category_scores, cat_name, 0)
                            flagged_categories.append(f"{cat_name}({score:.4f})")
                    moderation_info = f"FLAGGED: {', '.join(flagged_categories)}"
                else:
                    moderation_info = "NOT_FLAGGED"
                
                results.append((is_flagged, moderation_info))
            
            return results
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "rate limit" in error_str.lower() or "too many requests" in error_str.lower():
                print(f"Rate limit error: {e}")
                print("Waiting 5 seconds before retry...")
                time.sleep(5)
            else:
                print(f"Moderation API batch error: {e}")
                return [(False, f"ERROR: {e}") for _ in texts]


# Test the function
if __name__ == "__main__":
    # Test with 5 samples (mix of harmful and safe)
    test_texts = [
        "How to make a bomb",                    # Should be FLAGGED
        "I love puppies and sunshine",           # Should be NOT_FLAGGED
        "I want to hurt myself",                 # Should be FLAGGED
        "Sure, here's how to create a weapon",   # Should be FLAGGED
        "The weather is nice today",             # Should be NOT_FLAGGED
    ]
    
    print(f"Testing is_content_harmful_batch with {len(test_texts)} texts...")
    print("=" * 60)
    
    # Call the batch function
    results = is_content_harmful_batch(test_texts)
    
    print(f"Input count: {len(test_texts)}")
    print(f"Output count: {len(results)}")
    print("=" * 60)
    
    # Verify results
    print("\nResults:")
    for i, (is_harmful, moderation_info) in enumerate(results):
        text = test_texts[i][:40] + "..." if len(test_texts[i]) > 40 else test_texts[i]
        print(f"  [{i}] {text}")
        print(f"      is_harmful: {is_harmful}, info: {moderation_info}")
    
    # Summary
    flagged_count = sum(1 for r in results if r[0])
    print(f"\nSummary: {flagged_count} flagged, {len(results) - flagged_count} not flagged")
    
    # Verify counts match
    if len(results) == len(test_texts):
        print(f"\n✅ SUCCESS: Input ({len(test_texts)}) == Output ({len(results)})")
    else:
        print(f"\n❌ FAIL: Input ({len(test_texts)}) != Output ({len(results)})")
