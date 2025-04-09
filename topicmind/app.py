import os
import logging
import nltk
from flask import Flask, request, jsonify
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Import TopicMind Components ---
# Ensure utils and models are importable (e.g., by running from the root directory or setting PYTHONPATH)
try:
    from utils.preprocessor import clean_text
    from models.lda_topic_model import load_lda_model_and_dict, get_document_topics, get_topic_top_words
    from utils.topic_refiner import refine_topic_name # Assumes OPENAI_API_KEY is set in environment
    from models.bart_summarizer import load_summarizer_model, summarize_text
except ImportError as e:
    logging.error(f"Error importing TopicMind components: {e}. Ensure PYTHONPATH is set or run from project root.")
    # Exit or handle gracefully if components are missing
    exit(1)

# --- Download NLTK data (if needed) ---
# User needs to run this once manually or integrate into setup
# try:
#     nltk.data.find('tokenizers/punkt')
# except nltk.downloader.DownloadError:
#     logging.info("NLTK 'punkt' tokenizer not found. Downloading...")
#     nltk.download('punkt')
#     logging.info("NLTK 'punkt' downloaded successfully.")

app = Flask(__name__)

# --- Load Models on Startup ---
lda_model, dictionary = None, None
summarizer_model_loaded = False

@app.before_first_request
def load_models():
    global lda_model, dictionary, summarizer_model_loaded
    logging.info("Loading models...")
    # Load LDA Model
    lda_model, dictionary = load_lda_model_and_dict() # Uses default paths in lda_topic_model.py
    if not lda_model or not dictionary:
        logging.warning("LDA model/dictionary failed to load. Topic detection will not work.")
        # TODO: Consider triggering LDA training here if data is available

    # Load BART Summarizer
    load_summarizer_model() # This function handles its own logging/errors
    # We need a way to check if BART loaded successfully from bart_summarizer.py
    # Let's modify bart_summarizer to return a status or check its internal state.
    # For now, we assume it logs errors if it fails.
    # A simple check could be added to summarize_text, but it's better to know at startup.
    # ---> Placeholder: Assume load_summarizer_model indicates success/failure via logs for now.
    # ---> We will rely on the check within summarize_text for robustness.
    summarizer_model_loaded = True # Assume loaded unless error logged by the function
    logging.info("Model loading complete (check logs for errors).")

# --- API Endpoints ---
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    # Basic check, could be expanded to check model status
    model_status = {
        "lda_model_loaded": lda_model is not None and dictionary is not None,
        # "bart_model_loaded": Check if _model and _tokenizer in bart_summarizer are loaded (requires access or a status func)
    }
    return jsonify({"status": "ok", "models": model_status}), 200

@app.route('/analyze', methods=['POST'])
def analyze_text():
    """
    Endpoint to analyze text, extract topics, and generate summaries.
    Expects JSON input: {"text": "..."}
    Returns JSON output: {"results": [{"topic": "...", "summary": "..."}, ...]} or {"error": "..."}
    """
    if not request.json or 'text' not in request.json:
        logging.warning("Request received without 'text' field.")
        return jsonify({"error": "Missing 'text' in request body"}), 400

    input_text = request.json['text']

    if not input_text or not input_text.strip():
        logging.warning("Request received with empty 'text' field.")
        return jsonify({"error": "Input text cannot be empty"}), 400

    # Check if models are loaded
    if not lda_model or not dictionary:
        logging.error("LDA model not loaded. Cannot process request.")
        return jsonify({"error": "LDA model not available. Please check server logs."}), 503 # Service Unavailable
    # Add check for summarizer if a status check mechanism exists

    try:
        # 1. Preprocess text
        logging.info(f"Received text analysis request (length: {len(input_text)} chars).")
        cleaned_text = clean_text(input_text, remove_stopwords_flag=False) # Keep stopwords for sentence context
        logging.info("Text preprocessing complete.")

        # 2. Split into sentences
        try:
            sentences = nltk.sent_tokenize(cleaned_text)
        except LookupError:
            logging.error("NLTK 'punkt' tokenizer data not found. Cannot split into sentences.")
            return jsonify({"error": "Server configuration error: NLTK data missing."}), 500

        if not sentences:
            logging.info("No sentences found after preprocessing and tokenization.")
            return jsonify({"results": []}), 200 # Return empty results if no sentences

        logging.info(f"Split into {len(sentences)} sentences.")

        # 3. Assign topics to sentences
        sentence_topics = defaultdict(list) # {topic_id: [sentence1, sentence2, ...]}
        for sentence in sentences:
            if not sentence.strip(): continue # Skip empty sentences
            # Get dominant topic for the sentence
            topic_distribution = get_document_topics(lda_model, dictionary, sentence)
            if topic_distribution: # If topics were found
                dominant_topic_id = topic_distribution[0][0] # Get ID of the most probable topic
                sentence_topics[dominant_topic_id].append(sentence)
            # else: Log sentences that couldn't be assigned a topic?

        logging.info(f"Assigned sentences to {len(sentence_topics)} potential topics.")

        # 4. Refine topics and Summarize
        results = []
        if not sentence_topics:
            logging.info("No dominant topics identified for any sentence.")
            return jsonify({"results": []}), 200

        for topic_id, topic_sentences in sentence_topics.items():
            logging.info(f"Processing Topic ID: {topic_id} ({len(topic_sentences)} sentences)")
            # Get keywords for refinement
            top_words_probs = get_topic_top_words(lda_model, topic_id, num_words=10)
            keywords = [word for word, prob in top_words_probs]

            if not keywords:
                logging.warning(f"Could not get keywords for Topic ID: {topic_id}. Skipping refinement.")
                refined_topic_name = f"Topic {topic_id}" # Fallback name
            else:
                # Refine topic name via OpenAI
                refined_topic_name = refine_topic_name(keywords)
                logging.info(f"Refined Topic ID {topic_id} (Keywords: {keywords}) to: '{refined_topic_name}'.")

            # Summarize sentences for this topic
            # Join sentences into a single block for summarization
            text_to_summarize = " ".join(topic_sentences)
            summary = summarize_text(text_to_summarize, max_length=150, min_length=40) # Adjust lengths as needed

            # Check for summarization errors (e.g., if model wasn't loaded)
            if "Error:" in summary:
                logging.error(f"Summarization failed for Topic ID {topic_id}: {summary}")
                # Decide how to handle: skip topic, return error, return partial results?
                # For MVP, let's skip adding this topic if summarization failed critically.
                continue # Skip this topic if summary failed
            else:
                logging.info(f"Generated summary for Topic '{refined_topic_name}'.")

            results.append({"topic": refined_topic_name, "summary": summary})

        logging.info(f"Analysis complete. Returning {len(results)} topic-summary pairs.")
        return jsonify({"results": results}), 200

    except Exception as e:
        logging.exception(f"An unexpected error occurred during text analysis: {e}") # Log full traceback
        return jsonify({"error": "An internal error occurred during analysis."}), 500

if __name__ == '__main__':
    # Ensure models are loaded before running
    # Note: @app.before_first_request handles this, but explicit call might be needed
    # depending on WSGI server used if not using `flask run`
    # load_models() # Typically not needed with @before_first_request

    logging.info("Starting Flask server...")
    # Use waitress or gunicorn in production instead of Flask development server
    # Example: waitress-serve --host 0.0.0.0 --port 5001 topicmind.app:app
    app.run(host='0.0.0.0', port=5001, debug=False) # Set debug=False for production/testing load
