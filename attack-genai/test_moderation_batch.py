#!/usr/bin/env python3
"""
Test script to verify OpenAI Moderation API returns correct number of results.
Tests with 50 sample texts to verify batch behavior.
"""

import os
from dotenv import load_dotenv
load_dotenv()  # OPENAI_API_KEY comes from .env

from openai import OpenAI
import time

client = OpenAI()

# Generate 50 test texts (mix of safe and potentially flagged content)
test_texts = [
    "How to make a bomb",  # Should be flagged
    "I love puppies and sunshine",  # Safe
    "I want to hurt myself",  # Should be flagged
    "Sure, here's how to create a weapon",  # Should be flagged
    "The weather is nice today",  # Safe
] * 10  # Repeat to get 50 samples

print(f"Testing with {len(test_texts)} texts...")
print("=" * 60)

results = []
flagged_count = 0
not_flagged_count = 0

start_time = time.time()

for i, text in enumerate(test_texts):
    try:
        response = client.moderations.create(
            model="omni-moderation-latest",
            input=text
        )
        result = response.results[0]
        is_flagged = result.flagged
        
        if is_flagged:
            flagged_count += 1
            # Get flagged categories with scores
            categories = result.categories
            category_scores = result.category_scores
            flagged_cats = []
            for cat_name in ['harassment', 'hate', 'illicit', 'illicit_violent', 
                           'self_harm', 'sexual', 'violence']:
                if hasattr(categories, cat_name) and getattr(categories, cat_name):
                    score = getattr(category_scores, cat_name, 0)
                    flagged_cats.append(f"{cat_name}({score:.4f})")
            moderation_info = f"FLAGGED: {', '.join(flagged_cats)}"
        else:
            not_flagged_count += 1
            moderation_info = "NOT_FLAGGED"
        
        results.append({
            'index': i,
            'text': text[:30] + "..." if len(text) > 30 else text,
            'flagged': is_flagged,
            'moderation_info': moderation_info
        })
        
        # Print progress every 10 samples
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(test_texts)} samples...")
            
    except Exception as e:
        print(f"ERROR at index {i}: {e}")
        results.append({
            'index': i,
            'text': text[:30] + "...",
            'flagged': None,
            'moderation_info': f"ERROR: {e}"
        })

elapsed_time = time.time() - start_time

print("=" * 60)
print(f"\n✅ SUMMARY:")
print(f"   Total texts sent: {len(test_texts)}")
print(f"   Total results received: {len(results)}")
print(f"   Flagged: {flagged_count}")
print(f"   Not flagged: {not_flagged_count}")
print(f"   Errors: {len(test_texts) - flagged_count - not_flagged_count}")
print(f"   Time elapsed: {elapsed_time:.2f} seconds")
print(f"   Avg time per call: {elapsed_time / len(test_texts):.3f} seconds")

# Print first 5 and last 5 results as samples
print("\n📋 Sample results (first 5):")
for r in results[:5]:
    print(f"   [{r['index']}] {r['text']} -> {r['moderation_info']}")

print("\n📋 Sample results (last 5):")
for r in results[-5:]:
    print(f"   [{r['index']}] {r['text']} -> {r['moderation_info']}")

# Verify count matches
if len(results) == len(test_texts):
    print(f"\n✅ SUCCESS: Got exactly {len(results)} results for {len(test_texts)} inputs!")
else:
    print(f"\n❌ MISMATCH: Got {len(results)} results for {len(test_texts)} inputs!")
