#!/usr/bin/env python3
"""
Test script to verify environment variables and API connection
"""
import os
from dotenv import load_dotenv

def test_env_vars():
    """Test if environment variables are loaded correctly"""
    print("Testing environment variables...")
    
    # Load .env file
    load_dotenv()
    
    # Check if API key is loaded
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        print(f"✅ OPENAI_API_KEY loaded: {api_key[:10]}...{api_key[-4:]}")
    else:
        print("❌ OPENAI_API_KEY not found")
        return False
    
    return True

def test_simple_import():
    """Test if OpenAI package can be imported without issues"""
    try:
        from openai import AzureOpenAI
        print("✅ OpenAI package imported successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to import OpenAI: {e}")
        return False

def test_client_creation():
    """Test if Azure OpenAI client can be created"""
    try:
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        
        from openai import AzureOpenAI
        client = AzureOpenAI(
            azure_endpoint="https://rag-llm-api.openai.azure.com/",
            api_key=api_key,
            api_version="2025-01-01-preview"
        )
        print("✅ Azure OpenAI client created successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to create client: {e}")
        return False

if __name__ == "__main__":
    print("=== Environment and API Test ===\n")
    
    # Run tests
    env_ok = test_env_vars()
    import_ok = test_simple_import()
    client_ok = test_client_creation()
    
    print(f"\n=== Test Results ===")
    print(f"Environment Variables: {'✅' if env_ok else '❌'}")
    print(f"Package Import: {'✅' if import_ok else '❌'}")
    print(f"Client Creation: {'✅' if client_ok else '❌'}")
    
    if all([env_ok, import_ok, client_ok]):
        print("\n🎉 All tests passed! The setup should work correctly.")
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above.")
