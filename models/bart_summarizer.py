import os
from transformers import BartTokenizer, BartForConditionalGeneration
import torch
import logging

# Force CPU-only mode for Mac M1 compatibility
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Model name
MODEL_NAME = "facebook/bart-large-cnn"

# Global variables to hold the model and tokenizer (load once)
# This avoids reloading the model on every function call, which is slow.
_tokenizer = None
_model = None

def load_summarizer_model():
    """Loads the BART model and tokenizer. Call this once during application startup."""
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        logging.info(f"Loading BART model ({MODEL_NAME})... This may take a moment.")
        try:
            # Set default device to CPU for Mac M1 compatibility
            torch.set_default_device('cpu')
            
            _tokenizer = BartTokenizer.from_pretrained(MODEL_NAME)
            
            # Explicitly set device to CPU for Mac M1 compatibility
            device = torch.device("cpu")
            logging.info(f"Using device: {device} (forced for Mac M1 compatibility)")
            
            _model = BartForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)
            logging.info("BART model loaded successfully on CPU.")
        except Exception as e:
            logging.error(f"Error loading BART model: {e}")
            _tokenizer = None
            _model = None


def summarize_text(text: str | list[str], max_length: int = 100, min_length: int = 30, num_beams: int = 4) -> str:
    """
    Generates a summary for the given text(s) using the loaded BART model.

    Args:
        text: A single string or a list of strings (sentences/paragraphs) to summarize.
        max_length: The maximum length of the generated summary.
        min_length: The minimum length of the generated summary.
        num_beams: Number of beams for beam search. Higher values can improve quality but are slower.

    Returns:
        The generated summary string, or an error message if summarization fails.
    """
    if _tokenizer is None or _model is None:
        logging.error("Summarizer model not loaded. Call load_summarizer_model() first.")
        # Attempt to load now
        load_summarizer_model()
        if _tokenizer is None or _model is None: # Check again after attempting load
            return "Error: Summarizer model unavailable."

    # If input is a list of sentences, join them.
    if isinstance(text, list):
        input_text = " ".join(text)
    elif isinstance(text, str):
        input_text = text
    else:
        logging.warning(f"Invalid input type for summarization: {type(text)}. Expected str or list[str].")
        return "Error: Invalid input type for summarization."

    if not input_text.strip():
        logging.warning("Input text for summarization is empty.")
        return "Error: Empty input text."

    # If text is too short, don't summarize
    if len(input_text.split()) < min_length:
        logging.info(f"Text too short ({len(input_text.split())} words) for summarization. Returning original.")
        # Return a truncated version of the original text if it's very short
        return input_text[:max_length * 5] if len(input_text) > max_length * 5 else input_text

    try:
        # Ensure we're using the CPU device
        device = torch.device("cpu")

        # Prepare the input text
        inputs = _tokenizer([input_text], max_length=1024, return_tensors='pt', truncation=True).to(device)

        # Generate summary
        summary_ids = _model.generate(
            inputs['input_ids'],
            num_beams=num_beams,
            max_length=max_length,
            min_length=min_length,
            length_penalty=2.0, # Encourages longer summaries slightly
            early_stopping=True
        )

        # Decode the summary
        summary = _tokenizer.decode(summary_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return summary

    except Exception as e:
        logging.error(f"Error during text summarization: {e}")
        return f"Error: Failed to generate summary. {str(e)}"

# Example usage (optional)
# if __name__ == "__main__":
#     # Make sure to load the model first
#     load_summarizer_model()
#
#     if _model and _tokenizer: # Check if loading was successful
#         sample_text = (
#             "Reddit is a vast network of communities where people can dive into their interests, "
#             "hobbies, and passions. There's a community for whatever you're interested in. "
#             "Users submit content such as links, text posts, images, and videos, which are then voted up or down by other members. "
#             "Posts are organized by subject into user-created boards called \"communities\" or \"subreddits\". "
#             "Submissions with more upvotes appear towards the top of their subreddit and, if they receive enough upvotes, "
#             "ultimately on the site's front page. Despite the site's diversity, common themes emerge, "
#             "like discussions on technology, politics, gaming, and personal stories."
#         )
#         summary = summarize_text(sample_text, max_length=50, min_length=15)
#         print("\nOriginal Text:\n", sample_text)
#         print("\nGenerated Summary:\n", summary)
#
#         sample_sentences = [
#              "The company announced record profits for the last quarter.",
#              "Stock prices surged following the announcement.",
#              "Analysts predict continued growth in the tech sector."
#         ]
#         summary_list = summarize_text(sample_sentences, max_length=30, min_length=10)
#         print("\nOriginal Sentences:\n", sample_sentences)
#         print("\nGenerated Summary (from list):\n", summary_list)
#     else:
#         print("Skipping example usage as BART model failed to load.")
