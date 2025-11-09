import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
print(f"API Key found: {api_key[:10]}..." if api_key else "No API key found")

genai.configure(api_key=api_key)

print("\n=== Available Models ===")
try:
    for model in genai.list_models():
        if 'generateContent' in model.supported_generation_methods:
            print(f"✓ {model.name}")
except Exception as e:
    print(f"Error listing models: {e}")

print("\n=== Testing Model ===")
try:
    # Try different model names (updated for Gemini 2.0/2.5)
    test_models = [
        'models/gemini-2.5-flash',
        'models/gemini-2.0-flash',
        'models/gemini-flash-latest',
        'models/gemini-pro-latest',
        'models/gemini-2.5-pro'
    ]
    
    for model_name in test_models:
        try:
            print(f"\nTrying: {model_name}")
            model = genai.GenerativeModel(model_name)
            response = model.generate_content("Say hello")
            print(f"✓ SUCCESS with {model_name}")
            print(f"Response: {response.text[:100]}")
            break
        except Exception as e:
            print(f"✗ Failed: {str(e)[:100]}")
            
except Exception as e:
    print(f"Error: {e}")