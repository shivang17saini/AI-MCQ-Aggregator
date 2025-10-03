import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

print("=" * 60)
print("CHECKING AVAILABLE GEMINI MODELS")
print("=" * 60)

google_key = os.getenv("GOOGLE_API_KEY")

if not google_key:
    print("❌ GOOGLE_API_KEY not found!")
    exit(1)

try:
    genai.configure(api_key=google_key)
    print("\n✓ API Key configured\n")
    
    print("Available models that support generateContent:\n")
    
    for model in genai.list_models():
        if 'generateContent' in model.supported_generation_methods:
            print(f"  ✓ {model.name}")
            print(f"    Display Name: {model.display_name}")
            print(f"    Description: {model.description[:80]}...")
            print()
    
    print("\nTrying different model versions:")
    
    test_models = [
        'gemini-pro',
        'gemini-1.5-pro',
        'gemini-1.5-flash',
        'gemini-1.5-flash-latest',
        'gemini-1.5-pro-latest',
        'models/gemini-pro',
        'models/gemini-1.5-flash',
        'models/gemini-1.5-pro'
    ]
    
    for model_name in test_models:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content("Say hello")
            print(f"  ✅ '{model_name}' WORKS! Response: {response.text[:50]}")
            break
        except Exception as e:
            print(f"  ❌ '{model_name}' failed: {str(e)[:60]}")
    
except Exception as e:
    print(f"❌ Error: {e}")

print("\n" + "=" * 60)