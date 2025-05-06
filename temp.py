import diskcache

# Set your custom cache path here
cache = diskcache.Cache('/usa/taikun/07_transencoder/attack-genai')

# Print total entries
print(f"Total cache entries: {len(cache)}")

# Show first few keys and values (up to 5)
for i, key in enumerate(cache.iterkeys()):
    if i >= 5:
        break
    try:
        value = cache[key]
        print(f"\nKey {i+1}: {key}")
        print(f"  Type: {type(value)}")
        if isinstance(value, (str, int, float, list, dict)):
            preview = str(value)
            print("  Value Preview:", preview[:200] + "..." if len(preview) > 200 else preview)
        else:
            print("  (Not displaying full value for large object)")
    except Exception as e:
        print(f"  Error reading key: {e}")