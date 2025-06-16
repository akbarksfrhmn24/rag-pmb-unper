#!/usr/bin/env python3
"""
Test the updated LLM service
"""
import sys
import os

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.llm_service import query_openai

def test_llm_service():
    """Test the LLM service with a simple query"""
    print("Testing LLM service...")
    
    test_prompt = "Apa persyaratan masuk ke Universitas Perjuangan Tasikmalaya?"
    
    try:
        response = query_openai(test_prompt)
        print(f"✅ Response received:")
        print(f"Prompt: {test_prompt}")
        print(f"Response: {response[:200]}...")  # Show first 200 characters
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_llm_service()
    if success:
        print("\n🎉 LLM service test passed!")
    else:
        print("\n⚠️ LLM service test failed!")
