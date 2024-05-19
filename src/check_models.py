import openai
import os

# Set your OpenAI API key
openai.api_key = os.getenv("OPENAI_KEY")

# Function to list available models
def list_models():
    try:
        response = openai.Model.list()
        return [model['id'] for model in response['data']]
    except openai.error.OpenAIError as e:
        print(f"Error: {e}")
        return []

if __name__ == "__main__":
    # List the available models
    available_models = list_models()
    print(available_models)
