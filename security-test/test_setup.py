import os
import sys
from dotenv import load_dotenv

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
load_dotenv()

print("Testing setup for security comparison...")
print("="*50)

# Check for API keys
api_keys = {
    'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY'),
    'TOGETHER_API_KEY': os.getenv('TOGETHER_API_KEY'),
    'ANTHROPIC_API_KEY': os.getenv('ANTHROPIC_API_KEY'),
}

print("API Keys Status:")
for key_name, key_value in api_keys.items():
    if key_value:
        print(f"  {key_name}: Found (length: {len(key_value)})")
    else:
        print(f"  {key_name}: Not found")

# Check if OpenAI key exists
if not api_keys['OPENAI_API_KEY']:
    print("\nERROR: OPENAI_API_KEY not found. Please set it in your environment or .env file")
    sys.exit(1)

# Test imports
print("\nTesting imports...")
try:
    from minions.clients.openai import OpenAIClient
    print("  ✓ OpenAIClient imported successfully")
except Exception as e:
    print(f"  ✗ Failed to import OpenAIClient: {e}")

try:
    from minions.clients.ollama import OllamaClient
    print("  ✓ OllamaClient imported successfully")
except Exception as e:
    print(f"  ✗ Failed to import OllamaClient: {e}")

try:
    from minions.minions import Minions
    print("  ✓ Minions imported successfully")
except Exception as e:
    print(f"  ✗ Failed to import Minions: {e}")

# Test data file
print("\nChecking data file...")
data_path = "security-test/test_data_cleaned.csv"
if os.path.exists(data_path):
    import pandas as pd
    df = pd.read_csv(data_path)
    print(f"  ✓ Data file found: {len(df)} samples")
    print(f"    - Malicious: {(df['label_binary'] == True).sum()}")
    print(f"    - Benign: {(df['label_binary'] == False).sum()}")
else:
    print(f"  ✗ Data file not found at {data_path}")

# Test OpenAI connection
print("\nTesting OpenAI connection...")
try:
    client = OpenAIClient(model_name="gpt-4o-mini")
    messages = [{"role": "user", "content": "Say 'test successful' in 3 words"}]
    responses, usage = client.chat(messages)
    print(f"  ✓ OpenAI connection successful: {responses[0]}")
except Exception as e:
    print(f"  ✗ OpenAI connection failed: {e}")

# Test Ollama connection
print("\nTesting Ollama connection...")
try:
    client = OllamaClient(model_name="llama3.2")
    messages = [{"role": "user", "content": "Say 'test successful' in 3 words"}]
    responses, usage, done_reasons = client.chat(messages)
    print(f"  ✓ Ollama connection successful: {responses[0]}")
except Exception as e:
    print(f"  ✗ Ollama connection failed: {e}")
    print("  Note: Make sure Ollama is running with 'ollama serve'")

print("\n" + "="*50)
print("Setup test complete!") 