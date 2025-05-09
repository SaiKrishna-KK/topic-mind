import re
import string
import nltk
import logging
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Download nltk data if not present
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError as e:
    logging.info(f"NLTK data not found: {e}. Downloading...")
    nltk.download('punkt')
    nltk.download('stopwords')
    logging.info("NLTK data downloaded successfully.")

# Load stopwords once
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    logging.error("NLTK stopwords not found. Trying to download...")
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))


def clean_reddit_content(text: str) -> str:
    """
    Cleans Reddit-style post content by removing irrelevant elements like:
    - Upvote/downvote counts
    - Award/Share buttons
    - Timestamps (e.g., "3y ago")
    - Comment navigation elements
    - Username indicators
    
    Args:
        text: Raw Reddit post/comment content
        
    Returns:
        Cleaned content with Reddit UI elements removed
    """
    if not isinstance(text, str):
        return ""
    
    # Remove media player time indicators
    text = re.sub(r'\d+:\d+\s*/\s*\d+:\d+', '', text)
    
    # Remove UI indicators (like "Video", "Archived post", etc.)
    text = re.sub(r'Video|Archived post\.|New comments cannot be posted and votes cannot be cast\.', '', text)
    
    # Remove "Go to comments" line
    text = re.sub(r'Go to comments', '', text)
    
    # Remove Sort by: lines
    text = re.sub(r'Sort by:.*?\n', '', text)
    
    # Remove "Search Comments" and "Expand comment search" lines
    text = re.sub(r'Search Comments|Expand comment search', '', text)
    
    # Remove "Comments Section" line
    text = re.sub(r'Comments Section', '', text)
    
    # Remove upvote/downvote counts
    text = re.sub(r'Upvote\s*[\dKMk\.]*', '', text)
    text = re.sub(r'Downvote\s*[\dKMk\.]*', '', text)
    
    # Remove award/share buttons
    text = re.sub(r'Award|Share', '', text)
    
    # Remove username avatar lines and prefix
    text = re.sub(r'u/[a-zA-Z0-9_-]+ avatar', '', text)
    text = re.sub(r'u/[a-zA-Z0-9_-]+', '', text)
    
    # Remove timestamp indicators (e.g., "• 3y ago")
    text = re.sub(r'[•·]\s*\d+[ymwd]\s*ago', '', text)
    
    # Remove reply navigation (e.g., "15 more replies")
    text = re.sub(r'\d+\s*(more)?\s*replies', '', text)
    
    # Clean specific patterns in the example
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)  # Standalone numbers (like vote counts)
    
    # Clean double newlines and spaces
    text = re.sub(r'\n{2,}', '\n', text)
    text = re.sub(r' {2,}', ' ', text)
    
    return text.strip()


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
        logging.warning(f"Expected string input, got {type(text)}. Returning empty string.")
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

    # 7. Remove stopwords if requested
    if remove_stopwords_flag:
        try:
            tokens = word_tokenize(text)
            filtered_tokens = [word for word in tokens if word not in stop_words and word.isalpha()]
            text = ' '.join(filtered_tokens)
        except LookupError as e:
            logging.error(f"NLTK data error: {e}")
            logging.info("Trying to download required NLTK data...")
            nltk.download('punkt')
            tokens = word_tokenize(text)
            filtered_tokens = [word for word in tokens if word not in stop_words and word.isalpha()]
            text = ' '.join(filtered_tokens)
        except Exception as e:
            logging.error(f"Error during stopword removal: {e}")
            # Keep the original text if stopword removal fails

    # TODO: Consider lemmatization or stemming for better topic modeling if needed in the future

    return text

# For testing purposes
if __name__ == "__main__":
    sample_text = """
    Looks like a fun game
    Video

    0:00 / 0:59

    Archived post. New comments cannot be posted and votes cannot be cast.

    Upvote
    52K

    Downvote

    656
    Go to comments


    Share
    Share
    Sort by:

    Best

    Search Comments
    Expand comment search
    Comments Section
    hackertripz
    •
    3y ago
    Where is this?! So cool

    Upvote
    2.2K

    Downvote

    Award

    Share
    Share

    u/Gerrbear78 avatar
    Gerrbear78
    •
    3y ago
    We have one in Winnipeg, Manitoba Canada / company called Activate. Super cool, we love it. Many other game rooms too

    Upvote
    1.5K

    Downvote

    Award

    Share
    Share


    106 more replies
    u/jamescobalt avatar
    jamescobalt
    •
    3y ago
    This one is called Activate - they have locations in Winnipeg and Burlington (Canada) and Louisville and Gatlinburg (USA). There are similar "challenge arcade" venues - like Level99 (probably the biggest in North America) and Boda Borg in Massachusetts (and parts of Europe) and Time Zone in Rhode Island. I think there are a couple other arcades opening on the US west coast very soon. Russia has a number in mall locations but I'm not familiar with that market.
    """
    
    # Test Reddit content cleaning
    cleaned_reddit = clean_reddit_content(sample_text)
    print("CLEANED REDDIT CONTENT:\n", cleaned_reddit)
    
    # Test regular text cleaning
    cleaned = clean_text(cleaned_reddit, remove_stopwords_flag=False)
    print("\nFINAL CLEANED TEXT:\n", cleaned)
