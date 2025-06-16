"""
Alternative LLM service implementation with better error handling
"""
import os
import sys
from dotenv import load_dotenv

def init_client():
    """Initialize the OpenAI client with proper error handling"""
    try:
        # Load environment variables
        load_dotenv()
        
        # Get API key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        # Import and create client
        from openai import AzureOpenAI
        
        client = AzureOpenAI(
            azure_endpoint="https://rag-llm-api.openai.azure.com/",
            api_key=api_key,
            api_version="2025-01-01-preview"
        )
        
        return client
        
    except Exception as e:
        print(f"Failed to initialize OpenAI client: {e}")
        return None

def query_openai_safe(prompt: str) -> str:
    """
    Safe version of OpenAI query with extensive error handling
    """
    if not prompt or not prompt.strip():
        return "Error: Empty prompt provided"
    
    # Initialize client each time to avoid memory issues
    client = init_client()
    if not client:
        return "Error: Failed to initialize OpenAI client"
    
    try:
        # Limit prompt length
        max_length = 3000
        if len(prompt) > max_length:
            prompt = prompt[:max_length] + "..."
        
        # Make the API call with conservative settings
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Kamu adalah asisten AI yang membantu menjawab pertanyaan seputar PMB Universitas Perjuangan Tasikmalaya."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,  # Reduced token limit
            temperature=0.7,
            timeout=15  # Shorter timeout
        )
        
        if response and response.choices and len(response.choices) > 0:
            content = response.choices[0].message.content
            return content.strip() if content else "No response generated"
        else:
            return "Error: No response received from the model"
            
    except Exception as e:
        error_msg = str(e)
        print(f"API Error: {error_msg}")
        return f"Error: Failed to get response - {error_msg[:100]}..."
    
    finally:
        # Clean up
        client = None

if __name__ == "__main__":
    # Test the safe implementation
    test_prompt = "Apa itu PMB Unper?"
    result = query_openai_safe(test_prompt)
    print(f"Response: {result}")
