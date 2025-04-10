import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# TODO: Download nltk data if not present (run once)
# import nltk
# nltk.download('punkt')
# nltk.download('stopwords')

# Optional: Load stopwords once
# try:
#     stop_words = set(stopwords.words('english'))
# except LookupError:
#     print("NLTK stopwords not found. Downloading...")
#     import nltk
#     nltk.download('stopwords')
#     stop_words = set(stopwords.words('english'))


def clean_text(text: str, remove_stopwords_flag: bool = False) -> str:
    """
    Cleans the input text by removing URLs, emojis, markdown-like syntax,
    extra whitespace, and optionally stopwords. Normalizes to lowercase.

    Args:
        text: The raw input string.
        remove_stopwords_flag: If True, remove common English stopwords.

    Returns:
        The cleaned text string.
    """
    if not isinstance(text, str):
        # TODO: Add logging for type errors
        print(f"Warning: Expected string input, got {type(text)}. Returning empty string.")
        return ""

    # 1. Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # 2. Remove Emojis (basic range, might need refinement)
    # Reference: https://stackoverflow.com/questions/33404752/removing-emojis-from-a-string-in-python
    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)

    # 3. Remove Markdown (simple cases: *, _, `, [](), etc.)
    # Basic markdown removal (links handled by URL removal)
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # Bold **text**
    text = re.sub(r'\*([^*]+)\*', r'\1', text)      # Italic *text*
    text = re.sub(r'_([^_]+)_', r'\1', text)         # Italic _text_
    text = re.sub(r'`([^`]+)`', r'\1', text)         # Code `text`
    text = re.sub(r'\[([^\]]+)\]\([^)]*\)', r'\1', text) # Links [text](url) -> text
    text = re.sub(r'^>.*$', '', text, flags=re.MULTILINE) # Blockquotes > text
    text = re.sub(r'^[\*\-+] ', '', text, flags=re.MULTILINE) # List items * - +

    # 4. Normalize case to lowercase
    text = text.lower()

    # 5. Remove punctuation (optional, keep if needed for sentence structure)
    # text = text.translate(str.maketrans('', '', string.punctuation))
    # Consider keeping sentence-ending punctuation if tokenizing later.

    # 6. Remove extra whitespace and empty lines
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'\n+', '\n', text).strip() # Keep single newlines if paragraphs matter

    # 7. Optional: Remove stopwords
    # if remove_stopwords_flag:
    #     try:
    #         tokens = word_tokenize(text)
    #         filtered_tokens = [word for word in tokens if word not in stop_words and word.isalpha()] # Keep only alpha words
    #         text = ' '.join(filtered_tokens)
    #     except LookupError:
    #          print("NLTK 'punkt' tokenizer not found. Skipping stopword removal. Download with nltk.download('punkt')")
    #          # Fallback or re-raise could go here
    #     except Exception as e:
    #         print(f"Error during stopword removal: {e}") # Log error
    #         # Decide how to handle: return original text, partial result, or raise


    # TODO: Consider lemmatization or stemming for better topic modeling
    # from nltk.stem import WordNetLemmatizer
    # lemmatizer = WordNetLemmatizer()
    # tokens = word_tokenize(text)
    # text = ' '.join([lemmatizer.lemmatize(word) for word in tokens])

    return text

# Example usage (optional)
# if __name__ == "__main__":
#     sample_text = """
#     Here's a sample text with a URL: https://example.com and some **bold** text.
#     Also, an emoji 😊 and a list:
#     * Item 1
#     * Item 2
#     > A blockquote.
#     Check this `code`. It's pretty cool.
#     Another sentence for testing stopword removal if needed.
#     """
#     cleaned = clean_text(sample_text, remove_stopwords_flag=True)
#     print("Cleaned Text:\n", cleaned)
