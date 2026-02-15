import ollama
import json

print("Testing Ollama API response structure...\n")

try:
    response = ollama.list()
    print("Raw response type:", type(response))
    print("\nRaw response:")
    print(json.dumps(response, indent=2, default=str))
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()