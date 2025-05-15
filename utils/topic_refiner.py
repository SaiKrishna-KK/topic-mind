import os
import logging
from typing import Union, List, Optional
from openai import OpenAI
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables (especially OPENAI_API_KEY)
load_dotenv()

# Initialize OpenAI client
# It's good practice to handle the case where the key is missing
try:
    # Explicitly get the API key from environment 
    api_key = os.environ.get('OPENAI_API_KEY')
    
    if not api_key:
        logging.error("OPENAI_API_KEY environment variable is not set")
        client = None
    else:
        logging.info("Initializing OpenAI client with API key")
        client = OpenAI(api_key=api_key)
        # Verify the API key works with a simple call
        try:
            models = client.models.list()
            logging.info(f"OpenAI API connection successful: {len(models.data)} models available")
        except Exception as e:
            logging.error(f"API key verification failed: {e}")
            client = None
except Exception as e:
    logging.error(f"Failed to initialize OpenAI client: {e}")
    logging.error("Please ensure your OPENAI_API_KEY environment variable is set correctly.")
    client = None  # Indicate client is not available

PROMPT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'prompts')
DEFAULT_PROMPT_PATH = os.path.join(PROMPT_DIR, 'refine_topic.gpt.txt')

def load_prompt_template(prompt_path: str = DEFAULT_PROMPT_PATH) -> Optional[str]:
    """Loads the prompt template from the specified file."""
    try:
        with open(prompt_path, 'r') as f:
            return f.read()
    except FileNotFoundError:
        logging.error(f"Error: Prompt template file not found at {prompt_path}")
        return None
    except Exception as e:
        logging.error(f"Error loading prompt template: {e}")
        return None

def refine_topic_name(keywords: List[str], model: str = "gpt-3.5-turbo", prompt_template: Optional[str] = None) -> str:
    """
    Uses OpenAI's GPT model to generate a refined topic name from a list of keywords.

    Args:
        keywords: A list of keywords representing a topic.
        model: The OpenAI model to use (e.g., "gpt-3.5-turbo" is more widely available than "gpt-4-turbo-preview").
        prompt_template: The template string for the prompt. If None, loads default.

    Returns:
        A refined topic name string, or a default name like "Topic [keywords]" on failure.
    """
    if client is None:
        logging.warning("OpenAI client not initialized. Cannot refine topic name.")
        return f"Topic [{', '.join(keywords[:3])}...]"  # Fallback name

    if not keywords:
        return "Unknown Topic"

    if prompt_template is None:
        prompt_template = load_prompt_template()
        if prompt_template is None:
            # Fallback if template loading fails
            logging.warning("Could not load prompt template. Using fallback topic name.")
            return f"Topic [{', '.join(keywords[:3])}...]"

    # Format the prompt
    keyword_string = ", ".join(keywords)
    prompt = prompt_template.format(keywords=keyword_string)
    
    logging.info(f"Refining topic with keywords: {keyword_string}")

    try:
        # Call OpenAI API
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert in summarizing topics concisely."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=15,  # Keep response short
            temperature=0.2,  # Lower temperature for more deterministic output
            n=1
        )

        # Extract the refined name
        refined_name = response.choices[0].message.content.strip()
        logging.info(f"OpenAI refined name: {refined_name}")

        # Basic cleaning (remove quotes, ensure reasonable length)
        refined_name = refined_name.replace('"', '').replace("Topic Name:", "").strip()
        if len(refined_name.split()) > 5:  # Heuristic check for overly long names
            logging.warning(f"Refined topic name seems long: '{refined_name}'. Using first few words.")
            refined_name = " ".join(refined_name.split()[:3])  # Limit to ~3 words

        return refined_name if refined_name else f"Topic [{', '.join(keywords[:3])}...]"  # Fallback if empty response

    except Exception as e:
        logging.error(f"Error calling OpenAI API for topic refinement: {e}")
        # Fallback to a default name
        return f"Topic [{', '.join(keywords[:3])}...]"

# For testing - uncomment and run this file directly to test
if __name__ == "__main__":
    # Set these keywords based on expected BERTopic output
    sample_keywords = ['technology', 'ai', 'machine', 'learning', 'data']
    
    if client:  # Only run if client initialized successfully
        refined_name = refine_topic_name(sample_keywords)
        print(f"Keywords: {sample_keywords}")
        print(f"Refined Topic Name: {refined_name}")
        
        sample_keywords_2 = ['privacy', 'security', 'data', 'user', 'policy']
        refined_name_2 = refine_topic_name(sample_keywords_2)
        print(f"\nKeywords: {sample_keywords_2}")
        print(f"Refined Topic Name: {refined_name_2}")
    else:
        print("Skipping example usage as OpenAI client could not be initialized.")
        print("Check your OPENAI_API_KEY environment variable.")
