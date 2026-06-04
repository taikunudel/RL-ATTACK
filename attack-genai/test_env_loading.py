import os

print("--- Testing .env loading ---")
script_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(script_dir, '.env')
print(f"Expected .env path: {env_path}")
print(f"Exists? {os.path.exists(env_path)}")

if os.path.exists(env_path):
    print("Reading file content:")
    with open(env_path, 'r') as f:
        content = f.read()
        print(f"--- Content Start ---\n{content}\n--- Content End ---")
        
    print("\nParsing logic trace:")
    with open(env_path, 'r') as f:
        for line in f:
            print(f"Processing line: {repr(line)}")
            if line.strip() and not line.startswith('#') and '=' in line:
                key, value = line.strip().split('=', 1)
                print(f"  Key: {repr(key)}, Value (raw): {repr(value)}")
                if key == "OPENAI_API_KEY":
                    cleaned_value = value.strip().strip('"\'')
                    print(f"  Value (cleaned): {cleaned_value}")
                    os.environ[key] = cleaned_value
                    print("  -> Set env var!")
                    break

print(f"\nFinal check: os.environ.get('OPENAI_API_KEY') = {os.environ.get('OPENAI_API_KEY')[:10]}...")
