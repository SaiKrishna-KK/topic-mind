import os
import logging
import nltk
import sys
import time
import threading
from functools import wraps
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

# Load environment variables from .env file (if it exists)
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'), verbose=True)
# If .env file wasn't found, python-dotenv will print a message but won't throw an error

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
embedding_model_loaded = False

# Define a health state variable
health_status = {
    "ready": False,
    "message": "System initializing...",
    "models_loaded": False,
    "startup_time": time.time()
}

@app.before_request
def load_models():
    # Check if models are already loaded to avoid reloading on every request
    global summarizer_model_loaded, embedding_model_loaded
    if summarizer_model_loaded and embedding_model_loaded:
        return # Models already loaded

    logging.info("Loading models...")
    # Load BART Summarizer
    success, message = load_summarizer_model()  # Now returns tuple (success, message)
    if success:
        summarizer_model_loaded = True
        logging.info(f"BART Summarizer loaded successfully: {message}")
    else:
        logging.error(f"BART Summarizer failed to load: {message}")
        summarizer_model_loaded = False
    
    # Load SentenceTransformer for embeddings (caching it)
    try:
        from models.bart_summarizer import get_sentence_transformer
        embedding_model = get_sentence_transformer()
        if embedding_model:
            embedding_model_loaded = True
            logging.info("SentenceTransformer model loaded and cached.")
        else:
            logging.warning("SentenceTransformer model not loaded. Some features may not work optimally.")
    except Exception as e:
        logging.error(f"Error loading SentenceTransformer model: {e}")
    
    logging.info("Model loading complete (check logs for errors).")

# Add this after model initialization
def initialize_models_async():
    """Initialize models asynchronously to avoid blocking the API startup"""
    global health_status
    
    try:
        # Ensure directory exists for model cache
        os.makedirs("models/cache", exist_ok=True)
        
        # Check disk space
        import shutil
        disk = shutil.disk_usage(".")
        free_space_gb = disk.free / (1024**3)
        if free_space_gb < 1.0:
            health_status["message"] = f"Warning: Low disk space ({free_space_gb:.2f}GB free). Models may fail to load."
            logging.warning(health_status["message"])
        
        # Load models
        logging.info("Loading models...")
        from models.bart_summarizer import load_models
        success = load_models()
        
        if success:
            health_status["models_loaded"] = True
            health_status["ready"] = True
            health_status["message"] = "System ready"
            logging.info("All models loaded successfully!")
        else:
            health_status["message"] = "Some models failed to load, but system is operational"
            logging.warning(health_status["message"])
            health_status["ready"] = True  # Still operational even with partial models
    except Exception as e:
        logging.error(f"Error during model initialization: {str(e)}")
        health_status["message"] = f"Error initializing models: {str(e)}"
        # Still mark as ready so the API can still serve responses with appropriate error messages
        health_status["ready"] = True

# Start async initialization
threading.Thread(target=initialize_models_async, daemon=True).start()

# --- API Endpoints ---
@app.route('/health', methods=['GET'])
def health_check():
    """Enhanced health check endpoint that reports system status"""
    global health_status
    
    uptime = time.time() - health_status["startup_time"]
    
    response = {
        "status": "up" if health_status["ready"] else "initializing",
        "message": health_status["message"],
        "uptime_seconds": int(uptime),
        "models_loaded": health_status["models_loaded"]
    }
    
    # If not ready yet, return 503 Service Unavailable
    status_code = 200 if health_status["ready"] else 503
    
    return jsonify(response), status_code

@app.route('/analyze', methods=['POST'])
def analyze_text():
    """
    Endpoint to analyze text and generate topic-based summaries.
    Expects JSON input: {"text": "...", "is_reddit_content": false, "num_topics": 3}
    Returns JSON output: {"results": [{"topic": "Topic Name", "summary": "..."}]} or {"error": "..."}
    """
    if not request.json or 'text' not in request.json:
        logging.warning("Request received without 'text' field.")
        return jsonify({"error": "Missing 'text' in request body"}), 400

    input_text = request.json['text']
    is_reddit_content = request.json.get('is_reddit_content', False)  # Is this Reddit-style content?
    
    # Options for display in frontend
    show_pre_summary_sentences = request.json.get('show_pre_summary_sentences', False)
    show_chunk_summaries = request.json.get('show_chunk_summaries', False)
    
    # Dev mode option to limit processing
    dev_mode = request.json.get('dev_mode', False)
    
    # Topic settings
    num_topics = int(request.json.get('num_topics', 3))  # Default to 3 topics
    
    # Chunking and compression options
    chunked_summarization = request.json.get('chunked_summarization', True)
    final_compression = request.json.get('final_compression', True)
    chunk_size = int(request.json.get('chunk_size', 10))
    max_sentences = int(request.json.get('max_sentences_per_topic', 25))

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

        # 2. Extract sentences
        try:
            # Get all sentences from the text
            sentences = nltk.sent_tokenize(cleaned_text)
            
            # Filter out very short sentences (likely fragments)
            sentences = [s for s in sentences if len(s.split()) >= 5]
            
            # Limit to max_sentences for processing efficiency
            if len(sentences) > max_sentences * num_topics:
                logging.info(f"Limiting analysis to {max_sentences * num_topics} sentences (from {len(sentences)} total)")
                sentences = sentences[:max_sentences * num_topics]
                
            logging.info(f"Extracted {len(sentences)} sentences for analysis")
            
        except LookupError:
            logging.error("NLTK 'punkt' tokenizer data not found. Cannot split into sentences.")
            return jsonify({"error": "Server configuration error: NLTK data missing."}), 500

        # Results array for multiple topics
        results = []
        
        # If requesting multiple topics, use topic extraction
        if num_topics > 1:
            try:
                # Call topic analysis function (analyze_topics_in_text)
                topic_sentences, topic_keywords = analyze_topics_in_text(cleaned_text, num_topics=num_topics)
                
                # topic_sentences is a dict mapping topic IDs to sentence lists
                # topic_keywords is a dict mapping topic IDs to keyword lists
                logging.info(f"Extracted {len(topic_sentences)} topics")
                
                # Check if we got fewer topics than requested
                if len(topic_sentences) < num_topics:
                    topic_count_info = f"Found {len(topic_sentences)} topics instead of the requested {num_topics}"
                    logging.info(topic_count_info)
                
                # Process each topic
                for topic_id, sentences in topic_sentences.items():
                    # Get the keywords for this topic
                    words = topic_keywords.get(topic_id, [])
                    
                    # Only use topic_refiner if we have a valid OpenAI API key
                    if openai_api_key and openai_api_key != "your_openai_api_key_here":
                        try:
                            # Try to refine the topic name using OpenAI
                            refined_name = refine_topic_name(words)
                            topic_name = refined_name
                            logging.info(f"Using OpenAI refined topic name: {topic_name}")
                        except Exception as e:
                            # Fall back to simple format if refinement fails
                            topic_name = f"Topic: {', '.join(words[:3])}"
                            logging.warning(f"Topic refinement failed: {e}, using fallback name")
                    else:
                        # No valid API key, use simple format
                        topic_name = f"Topic: {', '.join(words[:3])}"
                        logging.info("No valid OpenAI API key, using keyword-based topic name")
                    
                    # Create result object for this topic
                    topic_result = {
                        "topic": topic_name,
                        "keywords": words
                    }
                    
                    # Add source sentences if requested
                    if show_pre_summary_sentences:
                        source_sentences = [{"text": sent, "source": f"t{topic_id}_s{i}"} 
                                          for i, sent in enumerate(sentences)]
                        topic_result["source_sentences"] = source_sentences
                    
                    # Summarize the topic using the same chunking logic
                    if sentences:
                        # Similar chunking and summarization as in the single-topic case
                        if chunked_summarization and len(sentences) > chunk_size:
                            chunks = [sentences[i:i + chunk_size] 
                                    for i in range(0, len(sentences), chunk_size)]
                            
                            # Limit chunks in dev mode
                            if dev_mode and len(chunks) > 3:
                                chunks = chunks[:3]
                                topic_result["dev_mode_limited"] = True
                            
                            # Add chunks to result if showing source sentences
                            if show_pre_summary_sentences:
                                chunk_data = []
                                for chunk_idx, chunk in enumerate(chunks):
                                    chunk_sentences = [{"text": sent, "source": f"t{topic_id}_s{i+chunk_idx*chunk_size}"} 
                                                    for i, sent in enumerate(chunk)]
                                    chunk_data.append(chunk_sentences)
                                topic_result["chunks"] = chunk_data
                            
                            # Summarize each chunk
                            chunk_summaries = []
                            for chunk_idx, chunk in enumerate(chunks):
                                chunk_text = " ".join(chunk)
                                chunk_summary = summarize_text(
                                    chunk_text, 
                                    max_length=100, 
                                    min_length=30,
                                    keywords=words,
                                    topic_name=topic_name,
                                    prompt_prefix=f"Summarize this content about {topic_name}:"
                                )
                                if not chunk_summary.startswith("Error:"):
                                    chunk_summaries.append(chunk_summary)
                            
                            # Add chunk summaries if requested
                            if show_chunk_summaries:
                                topic_result["chunk_summaries"] = chunk_summaries
                            
                            # Apply final compression if enabled
                            if final_compression and chunk_summaries:
                                combined_text = " ".join(chunk_summaries)
                                final_summary = summarize_text(
                                    combined_text, 
                                    max_length=150, 
                                    min_length=40,
                                    keywords=words,
                                    topic_name=topic_name,
                                    prompt_prefix=f"Create a coherent summary about {topic_name}:"
                                )
                                if not final_summary.startswith("Error:"):
                                    topic_result["summary"] = final_summary
                                    topic_result["final_compressed"] = True
                                else:
                                    topic_result["summary"] = " ".join(chunk_summaries)
                                    topic_result["final_compressed"] = False
                            else:
                                topic_result["summary"] = " ".join(chunk_summaries)
                                topic_result["final_compressed"] = False
                        else:
                            # No chunking, summarize all sentences together
                            text_to_summarize = " ".join(sentences)
                            summary = summarize_text(
                                text_to_summarize, 
                                max_length=150, 
                                min_length=40,
                                keywords=words,
                                topic_name=topic_name,
                                prompt_prefix=f"Provide a concise summary about {topic_name}:"
                            )
                            if not summary.startswith("Error:"):
                                topic_result["summary"] = summary
                            else:
                                topic_result["summary"] = f"Failed to summarize topic: {topic_name}"
                    else:
                        topic_result["summary"] = f"No sentences found for topic: {topic_name}"
                    
                    # Add this topic's result
                    results.append(topic_result)
                
                # Return results
                response_data = {"results": results}
                if len(results) < num_topics:
                    response_data["topic_count_info"] = f"Found {len(results)} topics instead of the requested {num_topics}"
                
                return jsonify(response_data), 200
                
            except Exception as e:
                logging.exception(f"Error in topic extraction: {e}")
                # Fall back to single topic mode if topic extraction fails
                logging.info("Falling back to single topic mode due to error")
                num_topics = 1
        
        # Single topic mode (either by request or fallback)
        if num_topics == 1 or not results:
            # 3. Generate a comprehensive summary
            # Prepare result object
            result = {
                "topic": "Complete Summary",
                "keywords": []  # We'll add keywords if available
            }
            
            # Add source sentences if requested
            if show_pre_summary_sentences:
                source_sentences = [{"text": sent, "source": f"s{i}"} for i, sent in enumerate(sentences)]
                result["source_sentences"] = source_sentences
            
            # Implement chunking if enabled and enough sentences
            if chunked_summarization and len(sentences) > chunk_size:
                # Split sentences into chunks
                chunks = [sentences[i:i + chunk_size] for i in range(0, len(sentences), chunk_size)]
                logging.info(f"Split {len(sentences)} sentences into {len(chunks)} chunks of max size {chunk_size}")
                
                # Limit the number of chunks in dev mode
                if dev_mode and len(chunks) > 3:
                    logging.info(f"DEV MODE: Limiting to first 3 chunks instead of {len(chunks)}")
                    chunks = chunks[:3]
                    # Inform the frontend that we limited processing
                    result["dev_mode_limited"] = True
                
                # Add chunks to result if showing source sentences
                if show_pre_summary_sentences:
                    # Format chunks with metadata
                    chunk_data = []
                    for chunk_idx, chunk in enumerate(chunks):
                        chunk_sentences = [{"text": sent, "source": f"s{i+chunk_idx*chunk_size}"} 
                                           for i, sent in enumerate(chunk)]
                        chunk_data.append(chunk_sentences)
                    result["chunks"] = chunk_data
                
                # Summarize each chunk
                chunk_summaries = []
                for chunk_idx, chunk in enumerate(chunks):
                    chunk_text = " ".join(chunk)
                    # Use a neutral prompt for summarization
                    chunk_summary = summarize_text(
                        chunk_text, 
                        max_length=100, 
                        min_length=30,
                        prompt_prefix="Summarize this text concisely:"
                    )
                    if not chunk_summary.startswith("Error:"):
                        chunk_summaries.append(chunk_summary)
                        logging.info(f"Generated summary for chunk {chunk_idx+1}/{len(chunks)}")
                
                # Add chunk summaries if requested
                if show_chunk_summaries:
                    result["chunk_summaries"] = chunk_summaries
                
                # Apply final compression if enabled
                if final_compression and chunk_summaries:
                    combined_text = " ".join(chunk_summaries)
                    # Use a neutral prompt for final summarization
                    final_summary = summarize_text(
                        combined_text, 
                        max_length=150, 
                        min_length=40,
                        prompt_prefix="Integrate these summaries into a coherent overview:"
                    )
                    if not final_summary.startswith("Error:"):
                        result["summary"] = final_summary
                        result["final_compressed"] = True
                        logging.info("Generated final compressed summary.")
                    else:
                        # No compression, just join chunk summaries
                        result["summary"] = " ".join(chunk_summaries)
                        result["final_compressed"] = False
                else:
                    # No compression, just join chunk summaries
                    result["summary"] = " ".join(chunk_summaries)
                    result["final_compressed"] = False
            else:
                # No chunking, summarize all sentences together
                text_to_summarize = " ".join(sentences)
                # Use a neutral prompt
                summary = summarize_text(
                    text_to_summarize, 
                    max_length=150, 
                    min_length=40,
                    prompt_prefix="Provide a concise summary of this text:"
                )
                
                # Check for summarization errors
                if not summary.startswith("Error:"):
                    result["summary"] = summary
                    logging.info("Generated direct summary.")
                else:
                    result["summary"] = "Summarization failed."
                    logging.error(f"Summarization failed: {summary}")
            
            # Return a single result in the results array for backward compatibility
            return jsonify({"results": [result]}), 200

        logging.info("Analysis complete. Returning summary.")

    except Exception as e:
        logging.exception(f"An unexpected error occurred during text analysis: {e}")  # Log full traceback
        return jsonify({"error": f"An internal error occurred during analysis: {str(e)}"}), 500

# Add a timeout wrapper for API endpoints
def timeout_handler(timeout_seconds=60):
    """Decorator to handle timeouts for API endpoints"""
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            # Get API timeout from environment or use default
            api_timeout = int(os.environ.get("API_TIMEOUT", timeout_seconds))
            
            def target():
                nonlocal result
                result = f(*args, **kwargs)
            
            result = None
            thread = threading.Thread(target=target)
            thread.daemon = True
            
            try:
                thread.start()
                thread.join(timeout=api_timeout)
                if thread.is_alive():
                    return jsonify({
                        "error": f"Request timed out after {api_timeout} seconds. The server might be under heavy load or processing large documents."
                    }), 504  # Gateway Timeout
                return result
            except Exception as e:
                return jsonify({
                    "error": f"An unexpected error occurred: {str(e)}"
                }), 500
        return wrapper
    return decorator

# Update the analyze_topics endpoint to use the timeout handler
@app.route("/analyze_topics", methods=["POST"])
@timeout_handler(60)  # 60 second timeout
def analyze_topics():
    """Analyze text and extract topics."""
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({"error": "No text provided"}), 400
            
        text = data['text']
        num_topics = data.get('num_topics', 5)  # Default to 5 topics
        
        # Ensure the number of topics is valid
        num_topics = max(1, min(10, num_topics))  # Between 1 and 10
        
        # Process the text to extract topics
        result = process_topics(text, num_topics)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

# Function to process topics from text
def process_topics(text, num_topics=5):
    """
    Process text to extract topics using topic modeling.
    
    Args:
        text: The text to analyze
        num_topics: Number of topics to extract
        
    Returns:
        Dictionary with topic information
    """
    try:
        # Clean the text
        cleaned_text = clean_text(text)
        
        # Use topic modeling to extract topics
        topic_sentences, topic_keywords = analyze_topics_in_text(cleaned_text, num_topics=num_topics)
        
        # Format results for API response
        results = []
        for topic_id, keywords in topic_keywords.items():
            # Try to refine the topic name if OpenAI API key is available
            if openai_api_key and openai_api_key != "your_openai_api_key_here":
                try:
                    topic_name = refine_topic_name(keywords)
                except Exception:
                    topic_name = f"Topic: {', '.join(keywords[:3])}"
            else:
                topic_name = f"Topic: {', '.join(keywords[:3])}"
                
            # Get sentences for this topic
            sentences = topic_sentences.get(topic_id, [])
            
            # Add to results
            results.append({
                "topic_id": topic_id,
                "topic_name": topic_name,
                "keywords": keywords,
                "sentence_count": len(sentences),
                "sentences": sentences[:5]  # Include a sample of sentences
            })
            
        return {
            "topic_count": len(results),
            "topics": results
        }

    except Exception as e:
        logging.exception(f"Error processing topics: {e}")
        return {"error": str(e)}

# Update the summarize endpoint to use the timeout handler
@app.route("/summarize", methods=["POST"])
@timeout_handler(60)  # 60 second timeout
def summarize():
    """Generate a summary of the provided text."""
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({"error": "No text provided"}), 400
        
        text = data['text']
        max_length = data.get('max_length', 150)
        min_length = data.get('min_length', 40)
        
        result = summarize_text(text, max_length=max_length, min_length=min_length)
        return jsonify({"summary": result})
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002)
