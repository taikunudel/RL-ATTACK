import diskcache
from train_attacker_genai import *

# Set your custom cache path here
cache = diskcache.Cache('/usa/taikun/rl-attack/attack-genai')

# Print total entries
print(f"Total cache entries: {len(cache)}")

# Show first few keys and values (up to 5)
for i, key in enumerate(cache.iterkeys()):
    if i >= 1:
        break
    try:
        value = cache[key]
        print(f"\nKey {i+1}: {key}")
        print(f"  Type: {type(value)}")
        if isinstance(value, (str, int, float, list, dict, tuple, np.ndarray)):  # Include np.ndarray and tuple
            if isinstance(value, np.ndarray):
                preview = str(value.tolist())  # Convert NumPy array to list for preview
            else:
                preview = str(value)
            print("  Value Preview:", preview[:200] + "..." if len(preview) > 200 else preview)
            # print("  Value Preview:", preview)
        else:
            print("  (Not displaying full value for large object)")
    except Exception as e:
        print(f"  Error reading key: {e}")

# # clean cash
# cache.clear()