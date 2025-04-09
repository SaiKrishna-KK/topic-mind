import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables (especially OPENAI_API_KEY)
load_dotenv()

# Initialize OpenAI client
# It's good practice to handle the case where the key is missing
try:
    client = OpenAI()
    # Attempt a simple API call to check authentication early if desired
    # client.models.list() # This verifies the key is valid
except Exception as e:
    print(f"Failed to initialize OpenAI client: {e}")
    print("Please ensure your OPENAI_API_KEY environment variable is set correctly.")
    client = None # Indicate client is not available

PROMPT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'prompts')
DEFAULT_PROMPT_PATH = os.path.join(PROMPT_DIR, 'refine_topic.gpt.txt')

def load_prompt_template(prompt_path: str = DEFAULT_PROMPT_PATH) -> str | None:
    """Loads the prompt template from the specified file."""
    try:
        with open(prompt_path, 'r') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: Prompt template file not found at {prompt_path}")
        return None
    except Exception as e:
        print(f"Error loading prompt template: {e}")
        return None

def refine_topic_name(keywords: list[str], model: str = "gpt-4-turbo-preview", prompt_template: str | None = None) -> str:
    """
    Uses OpenAI's GPT model to generate a refined topic name from a list of keywords.

    Args:
        keywords: A list of keywords representing a topic.
        model: The OpenAI model to use (e.g., "gpt-4-turbo-preview", "gpt-3.5-turbo").
        prompt_template: The template string for the prompt. If None, loads default.

    Returns:
        A refined topic name string, or a default name like "Topic [keywords]" on failure.
    """
    if client is None:
        print("OpenAI client not initialized. Cannot refine topic name.")
        return f"Topic [{', '.join(keywords[:3])}...]" # Fallback name

    if not keywords:
        return "Unknown Topic"

    if prompt_template is None:
        prompt_template = load_prompt_template()
        if prompt_template is None:
            # Fallback if template loading fails
            return f"Topic [{', '.join(keywords[:3])}...]"

    # Format the prompt
    keyword_string = ", ".join(keywords)
    prompt = prompt_template.format(keywords=keyword_string)

    try:
        # Call OpenAI API
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert in summarizing topics concisely."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=15, # Keep response short
            temperature=0.2, # Lower temperature for more deterministic output
            n=1,
            stop=None # Or perhaps ['\n'] if needed
        )

        # Extract the refined name
        refined_name = response.choices[0].message.content.strip()

        # Basic cleaning (remove quotes, ensure reasonable length)
        refined_name = refined_name.replace('"', '').replace("Topic Name:", "").strip()
        if len(refined_name.split()) > 5: # Heuristic check for overly long names
            print(f"Warning: Refined topic name seems long: '{refined_name}'. Using first few words.")
            refined_name = " ".join(refined_name.split()[:3]) # Limit to ~3 words

        return refined_name if refined_name else f"Topic [{', '.join(keywords[:3])}...]]" # Fallback if empty response

    except Exception as e:
        print(f"Error calling OpenAI API for topic refinement: {e}")
        # TODO: Implement more robust error handling (e.g., retries, logging)
        # Fallback to a default name
        return f"Topic [{', '.join(keywords[:3])}...]"

# Example usage (optional)
# if __name__ == "__main__":
#     sample_keywords = ['ai', 'machine', 'learning', 'model', 'data']
#     if client: # Only run if client initialized successfully
#         refined_name = refine_topic_name(sample_keywords)
#         print(f"Keywords: {sample_keywords}")
#         print(f"Refined Topic Name: {refined_name}")
#
#         sample_keywords_2 = ['privacy', 'security', 'data', 'user', 'policy']
#         refined_name_2 = refine_topic_name(sample_keywords_2)
#         print(f"\nKeywords: {sample_keywords_2}")
#         print(f"Refined Topic Name: {refined_name_2}")
#     else:
#         print("Skipping example usage as OpenAI client could not be initialized.")
