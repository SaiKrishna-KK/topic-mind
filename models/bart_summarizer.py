from transformers import BartTokenizer, BartForConditionalGeneration
import torch

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
        print(f"Loading BART model ({MODEL_NAME})... This may take a moment.")
        try:
            _tokenizer = BartTokenizer.from_pretrained(MODEL_NAME)
            # Check for GPU availability
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Using device: {device}")
            _model = BartForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)
            print("BART model loaded successfully.")
        except Exception as e:
            print(f"Error loading BART model: {e}")
            # Handle error appropriately - maybe raise, maybe set models to None
            _tokenizer = None
            _model = None
            # TODO: Consider adding fallback mechanisms or clearer error propagation


def summarize_text(text: str | list[str], max_length: int = 100, min_length: int = 30, num_beams: int = 4) -> str:
    """
    Generates a summary for the given text(s) using the loaded BART model.

    Args:
        text: A single string or a list of strings (sentences/paragraphs) to summarize.
        max_length: The maximum length of the generated summary.
        min_length: The minimum length of the generated summary.
        num_beams: Number of beams for beam search. Higher values can improve quality but are slower.

    Returns:
        The generated summary string, or an empty string if summarization fails.
    """
    if _tokenizer is None or _model is None:
        print("Error: Summarizer model not loaded. Call load_summarizer_model() first.")
        # Attempt to load now? Or just fail?
        # load_summarizer_model() # Potentially load here, but might be slow on first request
        # if _tokenizer is None or _model is None: # Check again after attempting load
        return "Error: Summarizer model unavailable." # Return error message

    # If input is a list of sentences, join them.
    if isinstance(text, list):
        input_text = " ".join(text)
    elif isinstance(text, str):
        input_text = text
    else:
        print(f"Warning: Invalid input type for summarization: {type(text)}. Expected str or list[str].")
        return ""

    if not input_text.strip():
        print("Warning: Input text for summarization is empty.")
        return "" # Return empty string for empty input

    try:
        # Determine the device the model is on
        device = _model.device

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
        print(f"Error during text summarization: {e}")
        # TODO: Add more specific error handling (e.g., for OOM errors)
        return "Error generating summary."

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
