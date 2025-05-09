import os
import logging
import nltk
import sys
from flask import Flask, request, jsonify
from collections import defaultdict
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Force CPU-only mode for Mac M1 compatibility with models
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Check for dependencies
logging.info("Checking dependencies...")
try:
    import torch
    logging.info(f"PyTorch version {torch.__version__} available.")
except ImportError:
    logging.error("PyTorch not installed. Please install with: pip install torch")
    sys.exit(1)

try:
    import tensorflow as tf
    logging.info(f"TensorFlow version {tf.__version__} available.")
except ImportError:
    logging.warning("TensorFlow not installed. Some features may not work properly.")

try:
    from transformers import BartTokenizer, BartForConditionalGeneration
    logging.info("Hugging Face Transformers available for BART model.")
except ImportError:
    logging.error("Transformers not installed. Please install with: pip install transformers")
    sys.exit(1)

try:
    from sentence_transformers import SentenceTransformer
    logging.info("SentenceTransformers available for embedding generation.")
except ImportError:
    logging.error("SentenceTransformers not installed. Please install with: pip install sentence-transformers")
    sys.exit(1)

# Load environment variables from .env file
load_dotenv()

# Check for OpenAI API key and warn if missing
openai_api_key = os.environ.get('OPENAI_API_KEY', None)
if not openai_api_key or openai_api_key == "your_openai_api_key_here":
    print("*" * 80)
    print("WARNING: No valid OPENAI_API_KEY found in environment variables!")
    print("Topic refinement will fall back to generic names like 'Topic [keyword1, keyword2, ...]'")
    print("Set OPENAI_API_KEY in your environment or .env file.")
    print("*" * 80)

# --- Import TopicMind Components ---
# Ensure utils and models are importable (e.g., by running from the root directory or setting PYTHONPATH)
try:
    from utils.preprocessor import clean_text, clean_reddit_content
    # Use our simplified embedding model instead of BERTopic
    from models.bertopic_model_simple import analyze_topics_in_text, get_topic_top_words
    from utils.topic_refiner import refine_topic_name # Assumes OPENAI_API_KEY is set in environment
    from models.bart_summarizer import load_summarizer_model, summarize_text
    logging.info("Successfully imported all TopicMind components.")
except ImportError as e:
    logging.error(f"Error importing TopicMind components: {e}. Ensure PYTHONPATH is set or run from project root.")
    # Exit or handle gracefully if components are missing
    exit(1)

# --- Download NLTK data (if needed) ---
# User needs to run this once manually or integrate into setup
try:
    nltk.data.find('tokenizers/punkt')
    logging.info("NLTK punkt tokenizer already available.")
except nltk.downloader.DownloadError:
    logging.info("NLTK 'punkt' tokenizer not found. Downloading...")
    nltk.download('punkt')
    logging.info("NLTK 'punkt' downloaded successfully.")

app = Flask(__name__)

# --- Load Models on Startup ---
summarizer_model_loaded = False

@app.before_request
def load_models():
    # Check if models are already loaded to avoid reloading on every request
    global summarizer_model_loaded
    if summarizer_model_loaded:
        return # Models already loaded

    logging.info("Loading models...")
    # Load BART Summarizer
    load_summarizer_model() # This function handles its own logging/errors
    summarizer_model_loaded = True # Assume loaded unless error logged by the function
    logging.info("Model loading complete (check logs for errors).")

# --- API Endpoints ---
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    # Basic check, could be expanded to check model status
    model_status = {
        "summarizer_model_loaded": summarizer_model_loaded,
        "openai_api_key_set": bool(openai_api_key and openai_api_key != "your_openai_api_key_here")
    }
    return jsonify({"status": "ok", "models": model_status}), 200

@app.route('/analyze', methods=['POST'])
def analyze_text():
    """
    Endpoint to analyze text, extract topics, and generate summaries.
    Expects JSON input: {"text": "...", "num_topics": 5, "is_reddit_content": false}
    Returns JSON output: {"results": [{"topic": "...", "summary": "..."}, ...]} or {"error": "..."}
    """
    if not request.json or 'text' not in request.json:
        logging.warning("Request received without 'text' field.")
        return jsonify({"error": "Missing 'text' in request body"}), 400

    input_text = request.json['text']
    num_topics = int(request.json.get('num_topics', 5))  # Default to 5 topics if not specified
    is_reddit_content = request.json.get('is_reddit_content', False)  # Is this Reddit-style content?

    # Ensure num_topics is within reasonable range
    num_topics = max(1, min(num_topics, 10))  # Clamp between 1-10
    logging.info(f"Requested {num_topics} topics for analysis")

    if not input_text or not input_text.strip():
        logging.warning("Request received with empty 'text' field.")
        return jsonify({"error": "Input text cannot be empty"}), 400

    # Check for minimum text length
    min_text_length = 100  # Character minimum
    if len(input_text.strip()) < min_text_length:
        logging.warning(f"Text too short: {len(input_text.strip())} chars, minimum {min_text_length}")
        return jsonify({
            "error": f"Input text is too short ({len(input_text.strip())} chars). Please provide at least {min_text_length} characters for meaningful analysis.",
            "results": []
        }), 200

    try:
        # 1. Preprocess text
        logging.info(f"Received text analysis request (length: {len(input_text)} chars).")
        
        # First, clean Reddit-specific content if specified
        if is_reddit_content:
            logging.info("Applying Reddit-specific content cleaning")
            input_text = clean_reddit_content(input_text)
            
        # Then apply general text cleaning
        cleaned_text = clean_text(input_text, remove_stopwords_flag=False)  # Keep stopwords for sentence context
        logging.info("Text preprocessing complete.")

        # 2. Analyze topics
        try:
            # Use the new analyze_topics_in_text function which handles sentence splitting and topic modeling
            logging.info(f"Analyzing text with {num_topics} topics")
            sentence_topics, topic_keywords = analyze_topics_in_text(cleaned_text, num_topics=num_topics)
            
            if not sentence_topics:
                logging.info("No topics found in the text.")
                return jsonify({"results": []}), 200
                
            logging.info(f"Split into sentences and grouped into {len(sentence_topics)} topics")
            
        except LookupError:
            logging.error("NLTK 'punkt' tokenizer data not found. Cannot split into sentences.")
            return jsonify({"error": "Server configuration error: NLTK data missing."}), 500

        # 3. Refine topics and Summarize
        results = []
        for topic_id, topic_sentences in sentence_topics.items():
            logging.info(f"Processing Topic ID: {topic_id} ({len(topic_sentences)} sentences)")
            # Get keywords for refinement
            keywords = topic_keywords.get(topic_id, [])

            if not keywords:
                logging.warning(f"No keywords found for Topic ID: {topic_id}. Using default name.")
                refined_topic_name = f"Topic {topic_id}"  # Fallback name
            else:
                # Refine topic name via OpenAI
                refined_topic_name = refine_topic_name(keywords)
                logging.info(f"Refined Topic ID {topic_id} (Keywords: {keywords[:5]}) to: '{refined_topic_name}'.")

            # Summarize sentences for this topic
            # Join sentences into a single block for summarization
            text_to_summarize = " ".join(topic_sentences)
            summary = summarize_text(text_to_summarize, max_length=150, min_length=40)  # Adjust lengths as needed

            # Check for summarization errors
            if summary.startswith("Error:"):
                logging.error(f"Summarization failed for Topic ID {topic_id}: {summary}")
                continue  # Skip this topic if summary failed
            else:
                logging.info(f"Generated summary for Topic '{refined_topic_name}'.")

            results.append({
                "topic": refined_topic_name,
                "summary": summary,
                "keywords": keywords[:5]  # Include top 5 keywords in the response
            })

        logging.info(f"Analysis complete. Returning {len(results)} topic-summary pairs.")
        
        # If we got fewer topics than requested, include an explanation
        response_data = {"results": results}
        if results and len(results) < num_topics:
            response_data["topic_count_info"] = f"Found {len(results)} topics instead of the requested {num_topics}"
            logging.info(f"Note: Found {len(results)} topics instead of requested {num_topics}")
            
        return jsonify(response_data), 200

    except Exception as e:
        logging.exception(f"An unexpected error occurred during text analysis: {e}")  # Log full traceback
        return jsonify({"error": f"An internal error occurred during analysis: {str(e)}"}), 500

if __name__ == '__main__':
    logging.info("Starting Flask server...")
    # Use waitress or gunicorn in production instead of Flask development server
    app.run(host='0.0.0.0', port=5001, debug=False)  # Set debug=False for production/testing load
