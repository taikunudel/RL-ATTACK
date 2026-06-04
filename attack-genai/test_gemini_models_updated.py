
import os
import time
from gemini_moderation import get_gemini_client, GEMINI_MODERATION_PROMPT, GeminiModerationResponse

def test_gemini_models():
    print("Initializing Gemini Client...")
    try:
        # Load environment variables if needed (handled by get_gemini_client)
        client = get_gemini_client()
    except Exception as e:
        print(f"Failed to initialize client: {e}")
        return

    gemini_models = [
        "gemini-3-flash-preview",  # Try the new ones
        "gemini-2.5-flash", 
        "gemini-2.5-flash-lite",
        "gemini-2.5-pro",
        "gemini-2.0-flash", 
        "gemini-2.0-flash-lite",
        "gemini-1.5-flash", 
        "gemini-1.5-pro"
    ]

    print(f"\nTesting {len(gemini_models)} Gemini models...\n")
    print(f"{'Model Name':<30} | {'Status':<10} | {'Response Time':<15}")
    print("-" * 65)

    text_to_test = "I want to hurt myself." # Simple harmful text to trigger moderation

    for model_name in gemini_models:
        start_time = time.time()
        status = "UNKNOWN"
        try:
            prompt = GEMINI_MODERATION_PROMPT + f'"{text_to_test}"'
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_json_schema": GeminiModerationResponse.model_json_schema(),
                },
            )
            # Just check if we got a valid response text
            if response.text:
                status = "SUCCESS"
            else:
                status = "EMPTY"
                
        except Exception as e:
            error_msg = str(e)
            if "503" in error_msg:
                status = "503 OVERLOAD"
            elif "404" in error_msg:
                status = "404 NOT FOUND"
            elif "400" in error_msg:
                status = "400 BAD REQ"
            else:
                status = "ERROR" # str(e)[:10]

        duration = time.time() - start_time
        print(f"{model_name:<30} | {status:<10} | {duration:.2f}s")
        time.sleep(1) # Slight pause

if __name__ == "__main__":
    test_gemini_models()
