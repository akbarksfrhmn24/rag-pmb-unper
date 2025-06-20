from openai import AzureOpenAI
import os
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
        
        client = AzureOpenAI(
            azure_endpoint="https://rag-llm-api.openai.azure.com/",
            api_key=api_key,
            api_version="2025-01-01-preview"
        )
        
        return client
        
    except Exception as e:
        print(f"Failed to initialize OpenAI client: {e}")
        return None

def query_openai(prompt: str) -> str:
    """
    Query Azure OpenAI with the given prompt - Safe implementation
    
    Args:
        prompt (str): The user prompt to send to the model
        
    Returns:
        str: The response from the model
    """
    if not prompt or not prompt.strip():
        return "Error: Empty prompt provided"
    
    # Initialize client each time to avoid memory issues
    client = init_client()
    if not client:
        return "Error: Failed to initialize OpenAI client"
    
    try:
        # Limit prompt length to prevent memory issues
        max_prompt_length = 3000
        if len(prompt) > max_prompt_length:
            prompt = prompt[:max_prompt_length] + "..."
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Kamu adalah asisten AI yang membantu menjawab pertanyaan seputar Penerimaan Mahasiswa Baru (PMB) Universitas Perjuangan Tasikmalaya (Unper)."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=800,  # Conservative token limit
            temperature=0.7,
            timeout=20  # Reasonable timeout
        )
        
        if response and response.choices and len(response.choices) > 0:
            content = response.choices[0].message.content
            return content.strip() if content else "No response generated"
        else:
            return "Error: No response from the model"
            
    except Exception as e:
        error_msg = str(e)
        print(f"Error in query_openai: {error_msg}")
        return f"Error: Failed to get response from AI model - {error_msg[:100]}..."
    
    finally:
        # Clean up to prevent memory issues
        client = None
