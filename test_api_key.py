import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

# :white_check_mark: FORCE removal of GOOGLE_API_KEY
if "GOOGLE_API_KEY" in os.environ:
    print(":warning: Removing GOOGLE_API_KEY from environment...")
    del os.environ["GOOGLE_API_KEY"]

api_key = os.getenv("GEMINI_API_KEY")

print(f":key: API Key exists: {api_key is not None}")
print(f":key: API Key length: {len(api_key) if api_key else 0}")
print(f":key: API Key preview: {api_key[:20]}..." if api_key and len(api_key) > 20 else api_key)
print(f":mag: GOOGLE_API_KEY in env: {'GOOGLE_API_KEY' in os.environ}")

# Test the API key directly
try:
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents="Say hello"
    )
    print(":white_check_mark: API Key is VALID!")
    print(f"Response: {response.text}")
except Exception as e:
    print(f":x: API Key is INVALID or has issues:")
    print(f"Error: {e}")
