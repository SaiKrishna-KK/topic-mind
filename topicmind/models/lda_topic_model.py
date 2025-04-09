import gensim
from gensim import corpora
from gensim.models import LdaModel
from nltk.tokenize import word_tokenize
import pickle
import os
import json
import nltk

# Import preprocessor relative to the script's location when run directly
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..')) # Add parent directory (topicmind) to path
try:
    from utils.preprocessor import clean_text
except ImportError:
    print("Error: Could not import clean_text from utils.preprocessor.")
    print("Make sure you are running this script from the project root directory (e.g., 'python topicmind/models/lda_topic_model.py')")
    exit(1)

# Constants
DEFAULT_NUM_TOPICS = 5
DEFAULT_PASSES = 10
MODEL_DIR = os.path.dirname(__file__)
DEFAULT_MODEL_PATH = os.path.join(MODEL_DIR, 'lda_model.pkl')
DEFAULT_DICT_PATH = os.path.join(MODEL_DIR, 'lda_dictionary.pkl')

def train_lda_model(texts: list[str], num_topics: int = DEFAULT_NUM_TOPICS, passes: int = DEFAULT_PASSES, model_path: str = DEFAULT_MODEL_PATH, dictionary_path: str = DEFAULT_DICT_PATH):
    """
    Trains a Gensim LDA model on the provided texts and saves the model and dictionary.

    Args:
        texts: A list of documents (strings) for training.
        num_topics: The desired number of topics.
        passes: The number of passes through the corpus during training.
        model_path: Path to save the trained LDA model.
        dictionary_path: Path to save the Gensim dictionary.
    """
    # TODO: Ensure NLTK 'punkt' is downloaded if using word_tokenize here
    # try:
    #     tokenized_texts = [word_tokenize(clean_text(text)) for text in texts]
    # except LookupError:
    #     print("NLTK 'punkt' not found. Please run nltk.download('punkt')")
    #     # Handle error appropriately - maybe raise or exit
    #     return

    # Placeholder tokenization (assuming pre-cleaned text)
    tokenized_texts = [word_tokenize(text) for text in texts]

    # Create Dictionary and Corpus
    dictionary = corpora.Dictionary(tokenized_texts)
    # Optional: Filter extremes
    # dictionary.filter_extremes(no_below=5, no_above=0.5)
    corpus = [dictionary.doc2bow(text) for text in tokenized_texts]

    if not corpus or not any(corpus):
        print("Warning: Corpus is empty after tokenization/dictionary creation. Cannot train LDA model.")
        # Handle empty corpus scenario
        return

    print(f"Training LDA model with {num_topics} topics...")
    lda_model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        random_state=42, # For reproducibility
        update_every=1,
        chunksize=100,
        passes=passes,
        alpha='auto',
        per_word_topics=True
    )
    print("LDA model training complete.")

    # Save model and dictionary
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(lda_model, f)
    with open(dictionary_path, 'wb') as f:
        pickle.dump(dictionary, f)
    print(f"LDA model saved to {model_path}")
    print(f"Dictionary saved to {dictionary_path}")


def load_lda_model_and_dict(model_path: str = DEFAULT_MODEL_PATH, dictionary_path: str = DEFAULT_DICT_PATH) -> tuple[LdaModel | None, corpora.Dictionary | None]:
    """
    Loads a previously saved LDA model and dictionary.

    Args:
        model_path: Path to the saved LDA model (.pkl).
        dictionary_path: Path to the saved Gensim dictionary (.pkl).

    Returns:
        A tuple containing the loaded (LdaModel, Dictionary), or (None, None) if not found.
    """
    lda_model = None
    dictionary = None
    try:
        with open(model_path, 'rb') as f:
            lda_model = pickle.load(f)
        with open(dictionary_path, 'rb') as f:
            dictionary = pickle.load(f)
        print(f"Loaded LDA model from {model_path} and dictionary from {dictionary_path}")
    except FileNotFoundError:
        print(f"Error: LDA model or dictionary file not found at specified paths.")
        # TODO: Consider triggering training if model doesn't exist or adding more robust error handling
    except Exception as e:
        print(f"Error loading LDA model or dictionary: {e}")
        # TODO: Add proper logging

    return lda_model, dictionary

def get_document_topics(lda_model: LdaModel, dictionary: corpora.Dictionary, document_text: str) -> list[tuple[int, float]]:
    """
    Infers the topic distribution for a single new document.

    Args:
        lda_model: The trained Gensim LDA model.
        dictionary: The Gensim dictionary used for training.
        document_text: The text of the document to analyze.

    Returns:
        A list of (topic_id, probability) tuples for the document.
    """
    # TODO: Ensure consistent preprocessing/tokenization with training
    # tokenized_doc = word_tokenize(clean_text(document_text))
    tokenized_doc = word_tokenize(document_text) # Placeholder

    # Convert document to BoW format
    bow_vector = dictionary.doc2bow(tokenized_doc)

    if not bow_vector:
        print("Warning: Document is empty after tokenization/BoW conversion. Cannot infer topics.")
        return []

    # Get topic distribution
    topic_distribution = lda_model.get_document_topics(bow_vector, minimum_probability=0.1) # Adjust minimum probability as needed
    return sorted(topic_distribution, key=lambda x: x[1], reverse=True)

def get_topic_top_words(lda_model: LdaModel, topic_id: int, num_words: int = 10) -> list[tuple[str, float]]:
    """
    Gets the top N words for a specific topic ID.

    Args:
        lda_model: The trained Gensim LDA model.
        topic_id: The ID of the topic.
        num_words: The number of top words to retrieve.

    Returns:
        A list of (word, probability) tuples.
    """
    try:
        top_words = lda_model.show_topic(topic_id, topn=num_words)
        return top_words
    except IndexError:
        print(f"Error: Topic ID {topic_id} is out of range.")
        return []
    except Exception as e:
        print(f"Error retrieving top words for topic {topic_id}: {e}")
        return []

# TODO: Implement logic to assign dominant topics to sentences/paragraphs
# This might involve splitting the input text, getting topics for each segment,
# and then grouping segments by their most probable topic.

# --- Training Script --- #
if __name__ == "__main__":
    # Configuration
    # Determine project root relative to this script
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    SAMPLE_DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'reddit_sample.json')
    NUM_TOPICS_FOR_SAMPLE = 3 # Adjust based on sample data size/diversity
    MODEL_SAVE_PATH = os.path.join(MODEL_DIR, 'lda_model.pkl')
    DICT_SAVE_PATH = os.path.join(MODEL_DIR, 'lda_dictionary.pkl')

    print(f"Looking for sample data at: {SAMPLE_DATA_PATH}")

    # 1. Load Sample Data
    try:
        with open(SAMPLE_DATA_PATH, 'r') as f:
            sample_data = json.load(f)
        # Extract text content
        training_docs = [item['text'] for item in sample_data if 'text' in item]
        if not training_docs:
            print(f"Error: No text found in {SAMPLE_DATA_PATH}. Cannot train LDA model.")
            exit(1)
        print(f"Loaded {len(training_docs)} documents from sample data.")
    except FileNotFoundError:
        print(f"Error: Sample data file not found at {SAMPLE_DATA_PATH}")
        print("Please ensure the file exists and contains sample JSON data.")
        exit(1)
    except json.JSONDecodeError:
        print(f"Error: Could not parse JSON from {SAMPLE_DATA_PATH}")
        exit(1)
    except Exception as e:
        print(f"An error occurred loading sample data: {e}")
        exit(1)

    # 2. Ensure NLTK data is available for tokenization
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords') # Needed if using stopwords in cleaning/tokenization
    except LookupError:
        print("NLTK 'punkt' or 'stopwords' data not found. Attempting download...")
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            print("NLTK data downloaded successfully.")
        except Exception as e:
            print(f"Failed to download NLTK data: {e}")
            print("Please install manually: run python -m nltk.downloader punkt stopwords")
            exit(1)

    # 3. Preprocess Data
    print("Preprocessing training documents...")
    # Apply cleaning (optional: enable stopword removal if desired for LDA)
    cleaned_docs = [clean_text(doc, remove_stopwords_flag=True) for doc in training_docs]
    # Filter out any empty documents that might result from cleaning
    cleaned_docs = [doc for doc in cleaned_docs if doc.strip()]
    if not cleaned_docs:
        print("Error: All documents became empty after preprocessing. Cannot train.")
        exit(1)
    print(f"Preprocessing complete. {len(cleaned_docs)} non-empty documents remaining.")

    # 4. Train and Save LDA Model
    print(f"Starting LDA model training with {NUM_TOPICS_FOR_SAMPLE} topics...")
    try:
        train_lda_model(
            texts=cleaned_docs,
            num_topics=NUM_TOPICS_FOR_SAMPLE,
            model_path=MODEL_SAVE_PATH,
            dictionary_path=DICT_SAVE_PATH
            # `train_lda_model` handles tokenization internally now
        )
        print("\n--- LDA Model Training Script Finished ---")
        print(f"Model saved to: {MODEL_SAVE_PATH}")
        print(f"Dictionary saved to: {DICT_SAVE_PATH}")
        print("You can now run the Flask backend.")

    except Exception as e:
        print(f"\n--- An error occurred during LDA model training --- ")
        print(e)
        # Consider adding more specific error handling if needed

# Example usage (optional - for testing this module)
# if __name__ == "__main__":
#     # Sample data (replace with actual data loading)
#     # Assume you have a list of documents called `training_docs`
#     training_docs = [
#         "Machine learning models require large datasets.",
#         "Deep learning is a subset of machine learning.",
#         "Natural language processing helps understand text.",
#         "Text summarization is an NLP task.",
#         "Data privacy is crucial in AI applications.",
#         "Ethical considerations are important for AI development."
#     ]
#
#     # 1. Train (if model doesn't exist)
#     if not os.path.exists(DEFAULT_MODEL_PATH):
#          print("Training model...")
#          # Preprocess your training_docs first!
#          # cleaned_docs = [clean_text(doc) for doc in training_docs]
#          train_lda_model(training_docs, num_topics=3)
#     else:
#          print("Model already exists. Skipping training.")
#
#     # 2. Load model
#     lda_model, dictionary = load_lda_model_and_dict()
#
#     if lda_model and dictionary:
#         # 3. Test inference on a new document
#         new_doc = "Artificial intelligence impacts society and privacy."
#         # cleaned_new_doc = clean_text(new_doc)
#         topics = get_document_topics(lda_model, dictionary, new_doc)
#         print(f"\nTopics for '{new_doc}': {topics}")
#
#         # 4. Show top words for each topic
#         num_topics_to_show = lda_model.num_topics
#         print("\nTop words per topic:")
#         for i in range(num_topics_to_show):
#             top_words = get_topic_top_words(lda_model, i, num_words=5)
#             print(f"Topic {i}: {top_words}") 