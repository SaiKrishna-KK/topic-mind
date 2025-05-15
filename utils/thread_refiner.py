import os
import re
import logging
import numpy as np
import hashlib
from typing import List, Dict, Tuple, Optional, Any, Union
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)
os.makedirs("logs/pipeline", exist_ok=True)

# Set up model constants
DEFAULT_MODEL_NAME = "all-mpnet-base-v2"

# Lists for filtering
NON_INFORMATIVE_REPLIES = [
    "lol", "same", "this", "haha", "yeah", "yep", "ok", "okay", "thx", "thanks", 
    "ty", "cool", "nice", "wow", "omg", "lmao", "rofl", "true", "false", "agree",
    "disagree", "upvoted", "downvoted", "saved", "true that", "exactly", "facts",
    "100%", "absolutely", "definitely", "for sure", "right"
]

SYMPATHY_ONLY_PHRASES = [
    "sorry to hear", "my condolences", "rip", "rest in peace", "thoughts and prayers",
    "sending love", "hugs", "that's terrible", "that's awful", "how sad",
    "so sad", "feel better", "get well soon", "thinking of you", "praying for you",
    "sorry for your loss", "sorry about that", "that sucks", "that's rough"
]

# Singleton model instance
_model = None

def load_embedding_model(model_name: str = DEFAULT_MODEL_NAME) -> Optional[SentenceTransformer]:
    """
    Loads the sentence transformer model for embeddings.
    
    Args:
        model_name: Name of the transformer model to load
    
    Returns:
        The loaded SentenceTransformer model, or None if loading fails
    """
    global _model
    if _model is not None:
        return _model
    
    try:
        logging.info(f"Loading sentence transformer model ({model_name}) in CPU-only mode")
        _model = SentenceTransformer(model_name, device='cpu')
        logging.info(f"Sentence transformer model loaded successfully")
        return _model
    except Exception as e:
        logging.error(f"Error loading sentence transformer model: {e}")
        return None

def is_non_informative(sentence: str) -> bool:
    """
    Checks if a sentence is a short, non-informative reply
    
    Args:
        sentence: The input sentence
        
    Returns:
        True if the sentence is non-informative, False otherwise
    """
    clean_text = sentence.strip().lower()
    
    # Check for very short sentences (less than 4 words and 15 chars)
    if len(clean_text.split()) < 4 and len(clean_text) < 15:
        return True
        
    # Check if sentence consists solely of a non-informative phrase
    for phrase in NON_INFORMATIVE_REPLIES:
        if clean_text == phrase or clean_text.startswith(phrase + " ") or clean_text.endswith(" " + phrase):
            return True
    
    return False

def is_sympathy_only(sentence: str) -> bool:
    """
    Checks if a sentence is only expressing sympathy without substantive content
    
    Args:
        sentence: The input sentence
        
    Returns:
        True if the sentence only expresses sympathy, False otherwise
    """
    clean_text = sentence.strip().lower()
    
    # Short sentences that contain sympathy phrases are likely sympathy-only
    if len(clean_text.split()) < 10:
        for phrase in SYMPATHY_ONLY_PHRASES:
            if phrase in clean_text:
                return True
    
    return False

def generate_sentence_hash(text: str) -> str:
    """
    Generate a unique identifier for a sentence
    
    Args:
        text: The sentence text
        
    Returns:
        A short hash string to uniquely identify the sentence
    """
    return hashlib.md5(text.encode('utf-8')).hexdigest()[:8]

def deduplicate_sentences(sentences: List[Dict[str, Any]], threshold: float = 0.85, model=None) -> List[Dict[str, Any]]:
    """
    Removes semantically similar sentences based on embedding similarity
    
    Args:
        sentences: List of sentence dictionaries to deduplicate
        threshold: Similarity threshold for deduplication (default: 0.85)
        model: Optional pre-loaded sentence transformer model
        
    Returns:
        Deduplicated list of sentence dictionaries
    """
    if not sentences:
        return []
        
    # Extract just the text for embedding
    sentence_texts = [s['text'] for s in sentences]
    
    # Load embedding model if not provided
    if model is None:
        model = load_embedding_model()
    
    if model is None:
        logger.error("Embedding model failed to load. Skipping deduplication.")
        return sentences
    
    # Get embeddings for all sentences
    try:
        embeddings = model.encode(sentence_texts, convert_to_tensor=True)
        embeddings = embeddings.cpu().numpy()  # Convert to numpy for similarity calculation
        
        # Calculate pairwise similarity
        similarity_matrix = cosine_similarity(embeddings)
        
        # Track indices to keep
        indices_to_keep = []
        for i in range(len(sentences)):
            # If this sentence is not similar to any previously kept sentence, keep it
            if not any(similarity_matrix[i, j] > threshold for j in indices_to_keep):
                indices_to_keep.append(i)
                
        # Return deduplicated sentences
        logger.info(f"Deduplicated from {len(sentences)} to {len(indices_to_keep)} sentences")
        return [sentences[i] for i in indices_to_keep]
        
    except Exception as e:
        logger.error(f"Error during sentence deduplication: {e}")
        return sentences

def rank_by_relevance(sentences: List[Dict[str, Any]], topic_keywords: List[str], max_sentences: int = 15) -> List[Dict[str, Any]]:
    """
    Ranks sentences by relevance to topic keywords using TF-IDF
    
    Args:
        sentences: List of sentence dictionaries to rank
        topic_keywords: List of topic keywords to compare against
        max_sentences: Maximum number of sentences to return
        
    Returns:
        List of sentence dictionaries ranked by relevance, limited to max_sentences
    """
    if not sentences or not topic_keywords:
        return sentences[:max_sentences] if sentences else []
    
    # Extract just the text for TF-IDF
    sentence_texts = [s['text'] for s in sentences]
    
    try:
        # Create a TF-IDF vectorizer
        tfidf = TfidfVectorizer(stop_words='english')
        
        # Fit TF-IDF on all sentences
        tfidf_matrix = tfidf.fit_transform(sentence_texts)
        
        # Create a keyword importance vector
        keywords_str = " ".join(topic_keywords)
        keywords_tfidf = tfidf.transform([keywords_str])
        
        # Get feature names
        feature_names = tfidf.get_feature_names_out()
        
        # Calculate sentence scores based on keywords
        sentence_scores = []
        
        for i, sentence_dict in enumerate(sentences):
            sentence = sentence_dict['text']
            # Get sentence vector
            sentence_vector = tfidf_matrix[i].toarray().flatten()
            
            # Calculate weighted score based on keyword importance
            score = 0
            
            # Boost score for sentences containing exact keywords
            for keyword in topic_keywords:
                if keyword.lower() in sentence.lower():
                    score += 3
            
            # Add TF-IDF score component
            for idx, word in enumerate(feature_names):
                if any(keyword.lower() in word.lower() or word.lower() in keyword.lower() 
                      for keyword in topic_keywords):
                    score += sentence_vector[idx] * 2  # Higher weight for keyword-related terms
                else:
                    score += sentence_vector[idx] * 0.5
                    
            sentence_scores.append((i, score, sentence_dict))
        
        # Sort sentences by score
        sentence_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Take top-k sentences
        top_sentences = [score[2] for score in sentence_scores[:max_sentences]]
        
        # Add the relevance score to each sentence dictionary
        for i, (_, score, _) in enumerate(sentence_scores[:max_sentences]):
            top_sentences[i]['relevance_score'] = float(score)
        
        # Log scoring results
        logger.info(f"Ranked {len(sentences)} sentences by relevance to {len(topic_keywords)} keywords")
        logger.info(f"Selected top {len(top_sentences)} most relevant sentences")
        
        return top_sentences
        
    except Exception as e:
        logger.error(f"Error during relevance ranking: {e}")
        return sentences[:max_sentences] if sentences else []

def refine_thread_content(text: str, topic_keywords: List[str], max_sentences: int = 15, 
                        model: Optional[SentenceTransformer] = None) -> List[Dict[str, Any]]:
    """
    Refines thread content by removing non-informative sentences, deduplicating similar content,
    and ranking by relevance to the topic keywords.
    
    Args:
        text: The thread text content (already cleaned from UI elements)
        topic_keywords: List of keywords for ranking relevance
        max_sentences: Maximum number of sentences to return
        model: Optional pre-loaded sentence transformer model
        
    Returns:
        List of refined and ranked sentence dictionaries with metadata
    """
    if not text or not text.strip():
        logger.warning("Empty text provided to refine_thread_content")
        return []
    
    # Tokenize into sentences
    try:
        raw_sentences = sent_tokenize(text)
    except LookupError:
        logger.info("NLTK punkt not found. Downloading...")
        nltk.download('punkt')
        raw_sentences = sent_tokenize(text)
    except Exception as e:
        logger.error(f"Error tokenizing sentences: {e}")
        return []
    
    logger.info(f"Split text into {len(raw_sentences)} sentences")
    
    # Convert to dictionary structure with provenance tracking
    sentences = []
    for i, sentence in enumerate(raw_sentences):
        if sentence.strip():  # Skip empty sentences
            sent_hash = generate_sentence_hash(sentence)
            sentences.append({
                'text': sentence,
                'source': f"s_{i}_{sent_hash}",
                'index': i,
                'hash': sent_hash
            })
    
    # Filter out non-informative replies
    sentences = [s for s in sentences if not is_non_informative(s['text'])]
    logger.info(f"Removed short non-informative replies. {len(sentences)} sentences remaining")
    
    # Filter out sympathy-only comments
    sentences = [s for s in sentences if not is_sympathy_only(s['text'])]
    logger.info(f"Removed sympathy-only comments. {len(sentences)} sentences remaining")
    
    # Deduplicate semantically similar sentences
    sentences = deduplicate_sentences(sentences, model=model)
    
    # Rank by relevance to topic keywords
    top_sentences = rank_by_relevance(sentences, topic_keywords, max_sentences)
    
    # Log the final result (text only, for simpler viewing)
    with open("logs/pipeline/semantic_cleaned_output.txt", "w") as f:
        f.write("\n\n".join([s['text'] for s in top_sentences]))
    
    # Log the full data with provenance
    with open("logs/pipeline/semantic_cleaned_with_provenance.json", "w") as f:
        import json
        json.dump(top_sentences, f, indent=2)
    
    return top_sentences

def chunk_sentences(sentences: List[Dict[str, Any]], chunk_size: int = 10) -> List[List[Dict[str, Any]]]:
    """
    Chunk sentences into groups for batch processing.
    
    Args:
        sentences: List of sentence dictionaries to chunk
        chunk_size: Number of sentences per chunk
        
    Returns:
        List of chunks, where each chunk is a list of sentence dictionaries
    """
    chunks = []
    
    for i in range(0, len(sentences), chunk_size):
        chunk = sentences[i:i+chunk_size]
        chunks.append(chunk)
        
    return chunks

# For backward compatibility with reddit_semantic_refiner
def refine_reddit_semantics(text: str, topic_keywords: List[str], max_sentences: int = 15,
                          model: Optional[SentenceTransformer] = None) -> List[Dict[str, Any]]:
    """
    Wrapper for refine_thread_content for Reddit content, maintained for backward compatibility.
    
    Args:
        text: The Reddit text content (already cleaned with clean_reddit_content)
        topic_keywords: List of keywords for ranking relevance
        max_sentences: Maximum number of sentences to return
        model: Optional pre-loaded sentence transformer model
        
    Returns:
        List of refined and ranked sentence dictionaries
    """
    return refine_thread_content(text, topic_keywords, max_sentences, model)

def process_thread_text(raw_text: str, topic_keywords: List[str], max_sentences: int = 15,
                      is_reddit: bool = False) -> List[Dict[str, Any]]:
    """
    Processes raw thread text by cleaning UI elements first (if Reddit), then applying
    semantic refinement to prioritize the most relevant content.
    
    Args:
        raw_text: Raw thread text, possibly with UI elements
        topic_keywords: List of keywords for relevance ranking
        max_sentences: Maximum number of final sentences to return
        is_reddit: Whether the content is from Reddit (requiring UI cleanup)
        
    Returns:
        List of processed and prioritized sentence dictionaries
    """
    # Clean the text if it's Reddit content
    if is_reddit:
        # Import here to avoid circular imports
        from utils.preprocessor import clean_reddit_content
        cleaned_text = clean_reddit_content(raw_text)
        logger.info("Cleaned Reddit UI elements from text")
    else:
        cleaned_text = raw_text
    
    # Apply semantic refinement
    refined_sentences = refine_thread_content(cleaned_text, topic_keywords, max_sentences)
    
    return refined_sentences

# For backward compatibility with reddit_semantic_refiner
def process_reddit_text(raw_text: str, topic_keywords: List[str], max_sentences: int = 15) -> List[Dict[str, Any]]:
    """
    Wrapper for process_thread_text specific to Reddit content, maintained for backward compatibility.
    
    Args:
        raw_text: Raw Reddit text with UI elements
        topic_keywords: List of keywords for relevance ranking
        max_sentences: Maximum number of final sentences to return
        
    Returns:
        List of processed and prioritized sentence dictionaries
    """
    return process_thread_text(raw_text, topic_keywords, max_sentences, is_reddit=True) 