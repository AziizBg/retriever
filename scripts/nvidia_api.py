from openai import OpenAI
from dotenv import load_dotenv
import os

def get_nvidia_client():
    """
    Creates and returns an OpenAI client configured for NVIDIA's API.
    
    Returns:
        OpenAI: Configured client for NVIDIA API calls
        
    Raises:
        ValueError: If NVIDIA_API_KEY is not found in environment variables
    """
    # Load environment variables from .env file
    load_dotenv()
    
    # Get the API key
    key = os.getenv("NVIDIA_API_KEY")
    if not key:
        raise ValueError("NVIDIA_API_KEY not found in environment variables")
    
    # Create and return the client
    return OpenAI(
        api_key=key,
        base_url="https://integrate.api.nvidia.com/v1"
    )

def call_nvidia_api(
    client,
    system_prompt,
    user_prompt,
    model_name="nvidia/llama-3.1-nemotron-ultra-253b-v1",
    temperature=0.2,
    max_tokens=2048
):
    """
    Makes a call to the NVIDIA API using the provided prompts and parameters.
    
    Args:
        client (OpenAI): The NVIDIA API client
        system_prompt (str): The system prompt to use
        user_prompt (str): The user prompt to use
        model_name (str, optional): The model to use. Defaults to "nvidia/llama-3.1-nemotron-ultra-253b-v1"
        temperature (float, optional): Controls randomness. Defaults to 0.2
        max_tokens (int, optional): Maximum tokens in response. Defaults to 2048
        
    Returns:
        str: The model's response text
        
    Raises:
        Exception: If the API call fails
    """
    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            top_p=1.0,
            max_tokens=max_tokens,
            frequency_penalty=0,
            presence_penalty=0,
            stream=False
        )
        
        return completion.choices[0].message.content.strip()
        
    except Exception as e:
        raise Exception(f"Error calling NVIDIA API: {str(e)}")

# Example usage:
if __name__ == "__main__":
    try:
        # Get the client
        client = get_nvidia_client()
        
        # Example prompts
        system_prompt = "You are a helpful assistant."
        user_prompt = "Hello, how are you?"
        
        # Make the API call
        response = call_nvidia_api(client, system_prompt, user_prompt)
        print("API Response:", response)
        
    except Exception as e:
        print(f"Error: {str(e)}") 