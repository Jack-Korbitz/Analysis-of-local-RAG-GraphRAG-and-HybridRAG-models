import ollama

def test_ollama_connection():
    """Test if Ollama service is running"""
    try:
        # List available models
        response = ollama.list()
        print("Ollama is running!")
        print("\nAvailable models:")
        
        # The response has a 'models' attribute with Model objects
        if hasattr(response, 'models'):
            for model in response.models:
                # Access the model attribute directly
                print(f"  - {model.model} ({model.details.parameter_size}, {model.details.quantization_level})")
        else:
            print("  (Unexpected response format)")
        
        return True
    except Exception as e:
        print(f"Ollama connection failed: {e}")
        print("\nMake sure Ollama is running:")
        print("  - Install: https://ollama.ai/download")
        print("  - Start: 'ollama serve' or check if it's running")
        return False

def test_model_inference(model_name="gemma2:27b"):
    """Test inference with a specific model"""
    try:
        print(f"\nTesting inference with {model_name}...")
        response = ollama.chat(
            model=model_name,
            messages=[{
                'role': 'user',
                'content': 'Say "Hello, I am working!" in exactly those words.'
            }]
        )
        
        result = response['message']['content']
        print(f"Model response: {result}")
        return True
    except Exception as e:
        print(f"Model inference failed: {e}")
        print(f"\nTry pulling the model first: ollama pull {model_name}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("OLLAMA CONNECTION TEST")
    print("=" * 60)
    
    # Test connection
    if test_ollama_connection():
        # Test your three models
        models_to_test = ["gemma3:12b", "gpt-oss:8b", "qwen3:8b"]
        
        print("\n" + "=" * 60)
        print("TESTING YOUR MODELS")
        print("=" * 60)
        
        for model in models_to_test:
            test_model_inference(model)
            print()