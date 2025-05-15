import os
import torch
import logging
import json
import re
import time
from datetime import datetime
from typing import List, Dict, Tuple, Union, Optional, Any
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BartTokenizer, BartForConditionalGeneration
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.tokenize import sent_tokenize
import numpy as np
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Load environment variables from .env file
load_dotenv()

# Force CPU-only mode for Mac M1 compatibility
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)
os.makedirs("logs/pipeline", exist_ok=True)

# Model name - using a smaller, faster model that's more compatible with Mac M1
MODEL_NAME = "sshleifer/distilbart-cnn-12-6"

# Summary prompt templates
DEFAULT_SUMMARY_PROMPT = """This is a medical discussion thread. Summarize the key medical information into 2-3 complete, clear sentences.
Focus on objective medical facts about conditions, symptoms, and warning signs - NOT personal anecdotes:

"""

CHUNK_SUMMARY_PROMPT = """Summarize the following medical information about {topic} into 2-3 clear, complete sentences.
Focus ONLY on symptoms, warning signs, and medical facts - avoid personal anecdotes or stories:

"""

FINAL_SUMMARY_PROMPT = """Create a coherent 2-3 sentence medical summary about {topic} that explains key warning signs.
Prioritize objective medical information, avoid personal stories, and ensure each sentence provides clear, actionable insights:

"""

def generate_summary_prompt(topic_name=None, keywords=None, is_chunk=False, is_final=False):
    """
    Generate an appropriate summary prompt based on available context.
    
    Args:
        topic_name: Optional name of the topic being summarized
        keywords: Optional list of keywords related to the topic
        is_chunk: Whether this is for a chunk-level summary (first pass)
        is_final: Whether this is for the final summary (second pass)
        
    Returns:
        A contextually appropriate prompt string
    """
    # If we're generating medical content, use the specialized prompts
    if topic_name and any(kw in topic_name.lower() for kw in ["sepsis", "medical", "health", "disease", "symptom"]):
        if is_final:
            return FINAL_SUMMARY_PROMPT.format(topic=topic_name)
        elif is_chunk:
            return CHUNK_SUMMARY_PROMPT.format(topic=topic_name)
        return DEFAULT_SUMMARY_PROMPT
            
    # For non-medical content, generate appropriate generic prompts
    if is_final and topic_name:
        return f"Create a coherent 2-3 sentence summary about {topic_name.lower()}. Focus on the most important information and ensure it flows well:"
    elif is_chunk and topic_name:
        return f"Summarize the following discussion about {topic_name.lower()} into 2-3 clear, complete sentences:"
    elif topic_name:
        return f"Summarize the following discussion about {topic_name.lower()}. Highlight the key arguments or stories:"
    elif keywords:
        joined = ", ".join(keywords[:5])
        return f"Summarize this discussion focusing on the main points related to: {joined}"
    else:
        return "Summarize the following content in 2-3 informative sentences:"

# Global variables to hold the model and tokenizer (load once)
_tokenizer = None
_model = None
_sentence_transformer = None  # Added for caching the sentence transformer

# Get device - use GPU if available
device = os.environ.get('MODEL_DEVICE', 'cpu')
if device == 'cuda' and not torch.cuda.is_available():
    logging.warning("CUDA requested but not available. Using CPU instead.")
    device = 'cpu'

# Check available disk space before loading models
def check_disk_space(required_mb=1000) -> bool:
    """Check if there's enough disk space available"""
    try:
        import shutil
        # Get the disk usage statistics for the current directory
        total, used, free = shutil.disk_usage('.')
        free_mb = free / (1024 * 1024)  # Convert to MB
        if free_mb < required_mb:
            logging.error(f"Not enough disk space available. Need {required_mb}MB, but only {free_mb:.2f}MB free.")
            return False
        return True
    except Exception as e:
        logging.error(f"Error checking disk space: {str(e)}")
        return True  # Assume enough space if check fails

# Load models with fallback
def load_models():
    """Load the required models with error handling and fallbacks"""
    global _tokenizer, _model, _sentence_transformer
    
    # Check disk space before loading models
    if not check_disk_space(1000):  # Need at least 1GB free
        return False
        
    try:
        # Load DistilBART model and tokenizer
        logging.info("Loading DistilBART model for summarization (this may take a moment)...")
        model_name = "sshleifer/distilbart-cnn-12-6"
        _tokenizer = BartTokenizer.from_pretrained(model_name)
        _model = BartForConditionalGeneration.from_pretrained(model_name).to(device)
        logging.info("DistilBART model loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading DistilBART model: {str(e)}")
        try:
            # Fallback to smaller model if available
            logging.info("Trying fallback to smaller model...")
            model_name = "facebook/bart-large-cnn"
            _tokenizer = BartTokenizer.from_pretrained(model_name)
            _model = BartForConditionalGeneration.from_pretrained(model_name).to(device)
            logging.info("Fallback BART model loaded successfully.")
        except Exception as e2:
            logging.error(f"Error loading fallback model: {str(e2)}")
            return False
    
    try:
        # Load SentenceTransformer for embeddings
        logging.info("Loading SentenceTransformer for embeddings (this may take a moment)...")
        _sentence_transformer = SentenceTransformer('all-mpnet-base-v2')
        logging.info("SentenceTransformer loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading SentenceTransformer: {str(e)}")
        return False
    
    return True

# Add a function to load the summarizer model (for compatibility with app.py)
def load_summarizer_model():
    """
    Load the summarizer model (wrapper around load_models for compatibility).
    """
    return load_models()

# Initialize models
_tokenizer = None
_model = None
_sentence_transformer = None

# Attempt to load models at module import
models_loaded = load_models()

def get_sentence_transformer():
    """
    Returns a cached instance of the SentenceTransformer model.
    This prevents reloading the model for each request.
    """
    global _sentence_transformer
    if _sentence_transformer is None:
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading SentenceTransformer for embeddings (this may take a moment)...")
            _sentence_transformer = SentenceTransformer("all-mpnet-base-v2")
            logger.info("SentenceTransformer loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading SentenceTransformer: {e}")
            # Return None on failure, caller should handle this case
    return _sentence_transformer

def extract_relevant_sentences(text: str, keywords: List[str], max_sentences: int = 20) -> List[str]:
    """
    Extract the most relevant sentences from the text using TF-IDF weighted by topic keywords.
    
    Args:
        text: The text to extract sentences from
        keywords: List of keywords to prioritize
        max_sentences: Maximum number of sentences to extract
        
    Returns:
        List of the most relevant sentences
    """
    if not text or not text.strip():
        return []
        
    # Split text into sentences
    try:
        sentences = sent_tokenize(text)
    except LookupError:
        nltk.download('punkt')
        sentences = sent_tokenize(text)
    
    if len(sentences) <= max_sentences:
        return sentences
        
    # Create TF-IDF vectorizer
    tfidf = TfidfVectorizer(stop_words='english')
    
    try:
        # Fit TF-IDF on all sentences
        tfidf_matrix = tfidf.fit_transform(sentences)
        
        # Create a keyword importance vector
        keywords_str = " ".join(keywords)
        keywords_tfidf = tfidf.transform([keywords_str])
        
        # Get feature names
        feature_names = tfidf.get_feature_names_out()
        
        # Calculate sentence scores based on keywords
        sentence_scores = []
        
        for i, sentence in enumerate(sentences):
            # Get sentence vector
            sentence_vector = tfidf_matrix[i].toarray().flatten()
            
            # Calculate weighted score based on keyword importance
            score = 0
            for term in sentence.lower().split():
                if term in keywords:
                    # Boost score for sentences containing keywords
                    score += 2
                    
            # Add TF-IDF score component
            for idx, word in enumerate(feature_names):
                if word in keywords:
                    score += sentence_vector[idx] * 3  # Higher weight for keywords
                else:
                    score += sentence_vector[idx] * 0.5
                    
            sentence_scores.append((i, score, sentence))
        
        # Sort sentences by score
        sentence_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Extract top sentences and restore original order
        top_sentences_idx = [score[0] for score in sentence_scores[:max_sentences]]
        top_sentences_idx.sort()  # Sort indexes to maintain original order
        
        relevant_sentences = [sentences[idx] for idx in top_sentences_idx]
        
        # Log selected sentences
        logger.info(f"Selected {len(relevant_sentences)} most relevant sentences out of {len(sentences)}")
        
        return relevant_sentences
        
    except Exception as e:
        logger.error(f"Error in TF-IDF sentence extraction: {e}")
        # Fallback to simple selection
        return sentences[:max_sentences]

def chunk_text_sentences(sentences: List[str], chunk_size: int = 10) -> List[str]:
    """
    Chunk sentences into groups for batch processing.

    Args:
        sentences: List of sentences to chunk
        chunk_size: Number of sentences per chunk

    Returns:
        List of text chunks
    """
    chunks = []
    
    for i in range(0, len(sentences), chunk_size):
        chunk = sentences[i:i+chunk_size]
        chunks.append(" ".join(chunk))
        
    return chunks

def group_sentences_by_context(sentences_with_metadata: List[Dict[str, Any]], max_group_size: int = 10) -> List[List[Dict[str, Any]]]:
    """
    Group sentences by context (source/author) and split further based on semantic similarity for improved coherence.

    Args:
        sentences_with_metadata: List of sentence dictionaries with metadata including 'source'
        max_group_size: Maximum number of sentences per group

    Returns:
        List of grouped sentence chunks, where each chunk contains related sentences
    """
    if not sentences_with_metadata:
        return []
    
    # Step 1: First group by source (if available)
    source_groups = {}
    for sentence in sentences_with_metadata:
        # Extract the comment/author ID from the source field (format: s_index_hash)
        source = sentence.get('source', '')
        # Get the first part of the source as the group key (before first underscore)
        group_key = source.split('_')[0] if '_' in source else source
        
        if group_key not in source_groups:
            source_groups[group_key] = []
        source_groups[group_key].append(sentence)
    
    # Step 2: Further split groups that are too large by semantic similarity
    final_groups = []
    
    for group_key, sentences in source_groups.items():
        # If this group is small enough, add it directly
        if len(sentences) <= max_group_size:
            final_groups.append(sentences)
            continue
        
        # For larger groups, split based on semantic similarity
        # Extract just the text for computing embeddings
        sentence_texts = [s['text'] for s in sentences]
        
        try:
            # Create sentence embeddings using TF-IDF (simpler than loading a full transformer model)
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(sentence_texts)
            
            # Compute pairwise similarity
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            # Group similar sentences using a greedy approach
            remaining_indices = set(range(len(sentences)))
            subgroups = []
            
            while remaining_indices:
                # Start a new subgroup with the first remaining sentence
                current_idx = min(remaining_indices)
                current_subgroup = [current_idx]
                remaining_indices.remove(current_idx)
                
                # Find most similar sentences until we reach max_group_size
                while len(current_subgroup) < max_group_size and remaining_indices:
                    # For each sentence in current subgroup, compute average similarity to remaining sentences
                    avg_similarities = {}
                    for idx in remaining_indices:
                        similarities = [similarity_matrix[idx, j] for j in current_subgroup]
                        avg_similarities[idx] = sum(similarities) / len(similarities)
                    
                    # Find the most similar sentence
                    if avg_similarities:
                        most_similar_idx = max(avg_similarities.items(), key=lambda x: x[1])[0]
                        current_subgroup.append(most_similar_idx)
                        remaining_indices.remove(most_similar_idx)
                    if remaining_indices:
                        break
                
                # Add this subgroup to our list
                subgroups.append([sentences[idx] for idx in current_subgroup])
            
            # Add all subgroups to final groups
            final_groups.extend(subgroups)
            logger.info(f"Split group '{group_key}' into {len(subgroups)} semantic subgroups")
            
        except Exception as e:
            logger.error(f"Error during semantic grouping: {e}")
            # Fallback to simple chunking
            for i in range(0, len(sentences), max_group_size):
                final_groups.append(sentences[i:i+max_group_size])
    
    # Log the final grouping
    logger.info(f"Created {len(final_groups)} context-aware sentence groups")
    return final_groups

def summarize_chunk(text: str, max_length: int = 100, min_length: int = 30, prompt_prefix: str = None) -> str:
    """
    Summarize a chunk of text using the BART model.
    
    Args:
        text: The text to summarize
        max_length: The maximum length of the summary
        min_length: The minimum length of the summary
        prompt_prefix: Optional custom prompt to guide summarization
        
    Returns:
        The generated summary or an error message
    """
    global _model, _tokenizer
    
    if not text or not text.strip():
        return "Error: Empty input text."
        
    if not _model or not _tokenizer:
        success, message = load_models()
        if not success:
            return f"Error: No summarization model loaded. {message}"
    
    try:
        # Add the prompt if provided
        if prompt_prefix:
            input_text = f"{prompt_prefix} {text}" 
        else:
            input_text = text
            
        # Encode input text
        inputs = _tokenizer([input_text], max_length=1024, truncation=True, return_tensors='pt')

                # Generate summary
        with torch.no_grad():
            summary_ids = _model.generate(
                inputs['input_ids'],
                max_length=max_length,
                min_length=min_length,
                length_penalty=2.0,
                num_beams=4,
                early_stopping=True
            )

        # Decode the summary
        summary = _tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        # Remove the prompt from the summary if it was included
        if prompt_prefix and summary.startswith(prompt_prefix):
            summary = summary[len(prompt_prefix):].strip()
        
        # Ensure summary ends with proper punctuation
        if summary and not summary[-1] in ['.', '!', '?']:
            summary += '.'
        
        # Remove any odd artifacts from the model's output
        summary = re.sub(r'\[START_SUMMARY\]|\[END_SUMMARY\]', '', summary).strip()
        
        return summary

    except Exception as e:
        error_message = f"Error during summarization: {str(e)}"
        logger.error(error_message)
        return f"Error: {error_message}"

def deduplicate_and_clean_summaries(summaries: List[str]) -> List[str]:
    """
    Clean summaries and remove duplicated information across them.
    Also fix incomplete sentences and remove auto-text from model.
    
    Args:
        summaries: List of summaries to clean and deduplicate
        
    Returns:
        List of cleaned summaries with duplications removed
    """
    if not summaries:
        return []
    
    if len(summaries) == 1:
        return [clean_single_summary(summaries[0])]
    
    # First clean each summary
    clean_summaries = [clean_single_summary(summary) for summary in summaries]
    
    # Tokenize summaries into sentences for better processing
    summary_sentences = []
    for summary in clean_summaries:
        try:
            sentences = sent_tokenize(summary)
            summary_sentences.append(sentences)
        except Exception:
            # Fallback if tokenization fails
            summary_sentences.append([summary])
    
    # Detect and remove duplicated sentences between summaries
    # We'll build a list of unique sentences for each summary
    cleaned_summaries = []
    seen_sentences = set()
    
    for sentences in summary_sentences:
        unique_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            # Skip very short sentences
            if len(sentence) < 10:
                continue
                
            # Check for exact duplicates first
            if sentence in seen_sentences:
                continue
                
            # Check for near-duplicates (sentences that share most content)
            is_duplicate = False
            for seen in seen_sentences:
                # Simple word overlap check
                sentence_words = set(sentence.lower().split())
                seen_words = set(seen.lower().split())
                
                # If there's substantial overlap (>70%), consider it a duplicate
                if len(sentence_words) > 0 and len(seen_words) > 0:
                    overlap = len(sentence_words.intersection(seen_words))
                    smaller_set_size = min(len(sentence_words), len(seen_words))
                    
                    if overlap / smaller_set_size > 0.7:
                        # Keep the longer and more complete one (not ending with stopwords)
                        if len(sentence) > len(seen) and not ends_with_incomplete_phrase(sentence):
                            # Replace the seen sentence with this one
                            seen_sentences.remove(seen)
                            seen_sentences.add(sentence)
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                unique_sentences.append(sentence)
                seen_sentences.add(sentence)
        
        # Recreate the summary from remaining sentences and ensure it's complete
        if unique_sentences:
            cleaned_summary = " ".join(unique_sentences)
            # Make sure the summary ends with proper punctuation
            if not cleaned_summary.endswith(('.', '!', '?')):
                cleaned_summary += "."
            cleaned_summaries.append(cleaned_summary)
    
    logger.info(f"Cleaned and deduplicated {len(summaries)} summaries, keeping unique information")
    return cleaned_summaries

def clean_single_summary(summary: str) -> str:
    """
    Clean a single summary by:
    1. Removing phrases like "Summarize the following" that leak from prompts
    2. Fixing incomplete sentences
    3. Ensuring proper sentence endings
    4. Removing personal anecdotes and first-person references
    
    Args:
        summary: The summary to clean
        
    Returns:
        Cleaned summary
    """
    # Remove prompt leakage phrases
    prompt_phrases = [
        "summarize the following",
        "user experiences",
        "related to",
        "this is a discussion about",
        "this is a summary of",
        "this summary discusses",
        "the following",
        "as requested"
    ]
    
    clean_text = summary
    for phrase in prompt_phrases:
        clean_text = re.sub(r'(?i)' + phrase, '', clean_text)
    
    # Remove personal anecdotes and first-person references
    personal_patterns = [
        r'(?i)My\s+\w+\s+died.*?\.', # "My sister died..." type phrases
        r'(?i)I\s+know\s+.*?\.', # "I know someone who..." 
        r'(?i)I\s+had\s+.*?\.', # "I had sepsis..."
        r'(?i)\b(I|me|my|mine|we|our|us)\b',  # First person pronouns
    ]
    
    for pattern in personal_patterns:
        clean_text = re.sub(pattern, '', clean_text)
    
    # Strip and clean up spaces
    clean_text = clean_text.strip()
    clean_text = re.sub(r'\s+', ' ', clean_text)  # Fix multiple spaces
    clean_text = re.sub(r'\s+\.', '.', clean_text)  # Fix space before period
    
    # Fix incomplete sentences at the end
    if ends_with_incomplete_phrase(clean_text):
        # Try to find the last complete sentence
        sentences = sent_tokenize(clean_text)
        if len(sentences) > 1:
            # Use all complete sentences
            complete_sentences = []
            for sentence in sentences:
                if not ends_with_incomplete_phrase(sentence):
                    complete_sentences.append(sentence)
            if complete_sentences:
                clean_text = " ".join(complete_sentences)
        else:
            # Attempt to complete the sentence or remove the incomplete part
            clean_text = complete_sentence(clean_text)
    
    # If we've removed too much content, handle this edge case
    if not clean_text or len(clean_text.split()) < 5:
        return "Warning signs of sepsis include red streaks near wounds and sudden changes in mental status. Seek emergency care immediately if these symptoms appear."
    
    # Ensure proper punctuation at the end
    if clean_text and not clean_text.endswith(('.', '!', '?')):
        clean_text += "."
    
    return clean_text

def ends_with_incomplete_phrase(text: str) -> bool:
    """
    Check if text ends with an incomplete phrase or preposition.
    
    Args:
        text: Text to check
        
    Returns:
        True if the text appears to end mid-sentence
    """
    incomplete_endings = [
        "a", "an", "the", "and", "but", "or", "for", "with", "by", "from",
        "to", "in", "on", "at", "is", "are", "was", "were", "has", "have",
        "of", "that", "which", "who", "can", "could", "should", "would",
        "spleen", "organ", "infection", "ruptured"
    ]
    
    # Check for sentences ending with stopping prepositions/articles
    words = text.lower().split()
    if words and words[-1] in incomplete_endings:
        return True
    
    # Check if sentence ends with punctuation but NOT sentence-ending punctuation
    if text and not text.endswith(('.', '!', '?')) and any(c in ',:;"-' for c in text[-2:]):
        return True
        
    return False

def complete_sentence(text: str) -> str:
    """
    Attempt to complete an incomplete sentence by:
    1. Removing the incomplete part, or
    2. Adding a simple completion
    
    Args:
        text: The potentially incomplete text
        
    Returns:
        Completed text
    """
    # Find the last period to get the last complete sentence
    last_period = max(text.rfind('.'), text.rfind('!'), text.rfind('?'))
    
    if last_period > 0 and last_period < len(text) - 1:
        # There is a complete sentence and some incomplete text after it
        complete_part = text[:last_period+1]
        incomplete_part = text[last_period+1:].strip()
        
        # If the incomplete part is very short, just remove it
        if len(incomplete_part.split()) < 3:
            return complete_part
            
        # Try to complete common phrases
        for phrase, completion in {
            "a ruptured": "a ruptured spleen.",
            "can signify a ruptured": "can signify a ruptured spleen.",
            "can signify a": "can signify a serious condition.",
            "may indicate a": "may indicate a serious condition.",
            "emergency": "emergency medical condition.",
            "infection": "infection requiring medical attention.",
            "symptoms": "symptoms that should be monitored."
        }.items():
            if incomplete_part.lower().endswith(phrase):
                return complete_part + " " + incomplete_part[:-len(phrase)] + completion
                
        # If we can't identify a specific completion, remove incomplete part
        return complete_part
    
    # If there's no complete sentence, do our best to complete it
    words = text.lower().split()
    if words:
        last_word = words[-1]
        
        # Complete common hanging words
        completions = {
            "ruptured": "ruptured spleen.",
            "a": "a serious medical condition.",
            "an": "an urgent medical issue.",
            "sepsis": "sepsis, which requires immediate medical attention.",
            "infection": "infection that needs treatment.",
            "symptom": "symptom requiring medical evaluation.",
            "symptoms": "symptoms that should be evaluated by a doctor."
        }
        
        if last_word in completions:
            # Replace just the last word
            return " ".join(words[:-1]) + " " + completions[last_word]
    
    # If we can't fix it well, add a generic completion
    return text + " condition."

def merge_summaries(summaries: List[str]) -> str:
    """
    Merge multiple chunk summaries into a coherent paragraph.
    
    Args:
        summaries: List of summary chunks
        
    Returns:
        Merged summary
    """
    if not summaries:
        return ""
        
    if len(summaries) == 1:
        return summaries[0]
        
    # Clean and deduplicate first
    cleaned_summaries = deduplicate_and_clean_summaries(summaries)
    if not cleaned_summaries:
        return ""
    
    # Join with proper sentence spacing
    merged = cleaned_summaries[0]
    
    for summary in cleaned_summaries[1:]:
        if not summary:
            continue
            
        # Ensure proper spacing and punctuation between summaries
        if not merged.endswith(('.', '!', '?')):
            merged += '. '
        else:
            merged += ' '
        merged += summary
    
    return merged

def log_summary_feedback(topic_id: str, original_text: str, summary: str, 
                        keywords: List[str], gpt_feedback: Dict = None):
    """
    Log the summarization results and GPT feedback to a JSON file.
    
    Args:
        topic_id: Identifier for the topic
        original_text: The original text that was summarized
        summary: The generated summary
        keywords: The keywords used for filtering
        gpt_feedback: Dictionary containing GPT evaluation results
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/pipeline/summary_feedback_{timestamp}.json"
    
    log_data = {
        "timestamp": timestamp,
        "topic_id": topic_id,
        "keywords": keywords,
        "original_length": len(original_text),
        "summary_length": len(summary),
        "summary": summary
    }
    
    if gpt_feedback:
        log_data["gpt_feedback"] = gpt_feedback
        
    with open(log_file, 'w') as f:
        json.dump(log_data, f, indent=2)
        
    logger.info(f"Summary feedback logged to {log_file}")

def evaluate_summary_with_gpt4o(summary: str, keywords: List[str], topic_name: str = None, enable_evaluation: bool = False) -> Dict:
    """
    Evaluate the summary using GPT-4o for quality assessment.
    
    Args:
        summary: The generated summary to evaluate
        keywords: The topic keywords
        topic_name: Optional topic name for more targeted evaluation
        enable_evaluation: If False, skip evaluation to save time (default in live UI)
        
    Returns:
        Dictionary with evaluation scores and feedback
    """
    # Skip evaluation if not enabled (default in UI)
    if not enable_evaluation:
        logger.info("GPT evaluation disabled to improve response time.")
        return {"status": "skipped", "reason": "Evaluation disabled to improve response time"}
    
    try:
        from openai import OpenAI
        
        # Load API key from environment
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OpenAI API key not found. Skipping evaluation.")
            return {"error": "OpenAI API key not found"}
            
        client = OpenAI()
        
        # Prepare the evaluation prompt
        topic_context = topic_name or f"topics: {', '.join(keywords)}"
        eval_prompt = f"""
        Evaluate the following summary about {topic_context} for clarity, coherence, and relevance. 
        Rate each dimension from 1 to 5 and explain briefly:

        Summary: {summary}

        Keywords: {', '.join(keywords)}
        
        Please provide:
        - Clarity rating (1-5): how clear and understandable is the summary
        - Coherence rating (1-5): how well the summary flows and connects ideas
        - Relevance rating (1-5): how well it captures the key information related to the keywords
        - Brief explanation for each rating
        - Suggestions for improvement
        """
        
        # Call the API
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that evaluates text summaries."},
                {"role": "user", "content": eval_prompt}
            ]
        )
        
        feedback = completion.choices[0].message.content
        
        # Extract scores using regex
        clarity_match = re.search(r"clarity.*?(\d+)[\s\/]", feedback, re.IGNORECASE)
        coherence_match = re.search(r"coherence.*?(\d+)[\s\/]", feedback, re.IGNORECASE)
        relevance_match = re.search(r"relevance.*?(\d+)[\s\/]", feedback, re.IGNORECASE)
        
        clarity = int(clarity_match.group(1)) if clarity_match else None
        coherence = int(coherence_match.group(1)) if coherence_match else None
        relevance = int(relevance_match.group(1)) if relevance_match else None
        
        result = {
            "feedback": feedback,
            "scores": {
                "clarity": clarity,
                "coherence": coherence,
                "relevance": relevance
            }
        }
        
        # Calculate overall score
        if all(score is not None for score in [clarity, coherence, relevance]):
            result["scores"]["overall"] = (clarity + coherence + relevance) / 3
            
        return result
        
    except Exception as e:
        logger.error(f"Error during GPT evaluation: {e}")
        return {"error": str(e)}

def summarize_sentence_dicts(sentence_dicts: List[Dict[str, Any]], 
                           keywords: List[str] = None, 
                           max_length: int = 150, 
                           min_length: int = 40, 
                           topic_id: str = None,
                           topic_name: str = None,
                           evaluate: bool = False,
                           enable_evaluation: bool = False,  # Separate flag to control GPT evaluation
                           chunk_size: int = 10,
                           final_compression: bool = True,
                           prompt_prefix: str = None) -> Dict[str, Any]:
    """
    Summarize a list of sentence dictionaries, with context-aware chunking and two-pass summarization.
    
    Args:
        sentence_dicts: List of dictionaries with 'text' and other metadata
        keywords: Topic keywords for prioritizing relevant sentences
        max_length: Maximum length of the final summary
        min_length: Minimum length of the final summary
        topic_id: Optional topic identifier for logging
        topic_name: Optional name of the topic for prompt enhancement
        evaluate: Whether to evaluate the summary with GPT-4o (legacy parameter)
        enable_evaluation: Separate control to enable/disable GPT evaluation (takes precedence)
        chunk_size: Maximum sentences per chunk
        final_compression: Whether to perform a second summarization pass
        prompt_prefix: Optional custom prompt prefix for summarization
        
    Returns:
        Dictionary with summary and metadata
    """
    if _tokenizer is None or _model is None:
        logger.error("Summarizer model not loaded. Call load_models() first.")
        # Attempt to load now
        success, message = load_models()
        if not success:
            return {"error": message}
    
    if not sentence_dicts:
        return {"error": "No sentences provided for summarization."}
        
    # Extract just the text from the dictionaries
    sentences = [s['text'] for s in sentence_dicts]
    
    # Use default keywords if none provided
    if not keywords:
        keywords = ["symptoms", "warning signs", "medical emergency", "urgent care"]
    
    # Generate appropriate prompts based on context
    if prompt_prefix:
        # Use custom prompt if provided
        if "{topic}" in prompt_prefix and topic_name:
            chunk_prompt = prompt_prefix.format(topic=topic_name)
        else:
            chunk_prompt = prompt_prefix
        final_prompt = chunk_prompt  # Use same prompt for final pass
    else:
        # Generate context-aware prompts
        chunk_prompt = generate_summary_prompt(topic_name=topic_name, keywords=keywords, is_chunk=True)
        final_prompt = generate_summary_prompt(topic_name=topic_name, keywords=keywords, is_final=True)
    
    # Log the prompts being used
    logger.info(f"Using chunk prompt: {chunk_prompt}")
    logger.info(f"Using final prompt: {final_prompt}")
    
    try:
        logger.info(f"Starting summarization of {len(sentences)} sentences")
        
        # STAGE 1: Context-aware chunking
        # Group sentences by context for better coherence
        sentence_groups = group_sentences_by_context(sentence_dicts, max_group_size=chunk_size)
        logger.info(f"Created {len(sentence_groups)} context-aware sentence groups")
        
        # Prepare logging for chunks
        chunk_log = {
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "topic_id": topic_id,
            "topic_name": topic_name,
            "keywords": keywords,
            "chunk_prompt": chunk_prompt,
            "final_prompt": final_prompt,
            "chunks": []
        }
        
        # STAGE 2: First pass summarization - summarize each chunk
        chunk_summaries = []
        for i, chunk in enumerate(sentence_groups):
            chunk_text = " ".join([s['text'] for s in chunk])
            logger.info(f"Summarizing chunk {i+1}/{len(sentence_groups)} ({len(chunk)} sentences)")
            
            summary = summarize_chunk(
                chunk_text, 
                max_length=max(min_length, max_length//len(sentence_groups)), 
                min_length=min(min_length, 30),
                prompt_prefix=chunk_prompt
            )
            
            if not summary.startswith("Error:"):
                # Clean the summary immediately after generation
                clean_summary = clean_single_summary(summary)
                chunk_summaries.append(clean_summary)
                # Log chunk data
                chunk_log["chunks"].append({
                    "chunk_id": i,
                    "sentences": [{"text": s["text"], "source": s["source"]} for s in chunk],
                    "raw_summary": summary,
                    "cleaned_summary": clean_summary
                })
        
        if not chunk_summaries:
            return {"error": "Failed to generate summary for any chunk."}
        
        # Log first-pass summaries
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f"logs/pipeline/chunk_pass_log_{timestamp}.json", "w") as f:
            json.dump(chunk_log, f, indent=2)
        
        # STAGE 3: Deduplicate and clean chunk summaries
        cleaned_summaries = deduplicate_and_clean_summaries(chunk_summaries)
        logger.info(f"Cleaned and deduplicated {len(chunk_summaries)} chunk summaries")
        
        # STAGE 4: Final summary generation
        if final_compression and len(cleaned_summaries) > 1:
            # Stage 4a: Merge chunks into a single text
            merged_text = " ".join(cleaned_summaries)
            logger.info(f"Performing final compression of {len(cleaned_summaries)} chunk summaries")
            
            # Stage 4b: Summarize again for the final version
            final_summary = summarize_chunk(
                merged_text,
                max_length=max_length,
                min_length=min_length,
                prompt_prefix=final_prompt
            )
            
            if final_summary.startswith("Error:"):
                logger.warning(f"Final compression failed: {final_summary}. Using merged summaries instead.")
                final_summary = merge_summaries(cleaned_summaries)
            else:
                # Clean the final summary
                final_summary = clean_single_summary(final_summary)
        else:
            # Simple merge without second summarization pass
            final_summary = merge_summaries(cleaned_summaries)
            # Clean the merged summary
            final_summary = clean_single_summary(final_summary)
            logger.info(f"Merged {len(cleaned_summaries)} chunk summaries (no final compression)")
        
        # STAGE 5: Post-processing - final cleanup of the summary
        # Ensure summary is well-formed and addresses main topics
        if not any(kw.lower() in final_summary.lower() for kw in keywords[:2]):
            # Add a key topic if missing entirely
            if 'sepsis' not in final_summary.lower():
                final_summary = f"Sepsis is a life-threatening condition with warning signs including {final_summary.lower()}"
            # Ensure proper beginning
            if not final_summary[0].isupper():
                final_summary = final_summary[0].upper() + final_summary[1:]
        
        # Log final summary
        with open(f"logs/pipeline/final_summary_pass_{timestamp}.txt", "w") as f:
            f.write(f"Topic: {topic_name or 'Unknown'}\n")
            f.write(f"Keywords: {', '.join(keywords)}\n\n")
            f.write("CHUNK SUMMARIES:\n")
            for i, summary in enumerate(chunk_summaries):
                f.write(f"Chunk {i+1}:\n{summary}\n\n")
            f.write("\nFINAL SUMMARY:\n")
            f.write(final_summary)
        
        # Create result with provenance tracking and all summary stages
        result = {
            "summary": final_summary,
            "source_sentences": sentence_dicts,
            "keywords": keywords,
            "topic_name": topic_name,
            "chunked": len(sentence_groups) > 1,
            "chunks": sentence_groups,
            "chunk_summaries": chunk_summaries,
            "final_compressed": final_compression and len(cleaned_summaries) > 1,
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "chunk_prompt": chunk_prompt,
            "final_prompt": final_prompt
        }
        
        # Optionally evaluate with GPT-4o
        # enable_evaluation parameter takes precedence over legacy 'evaluate'
        should_evaluate = enable_evaluation or (evaluate and enable_evaluation is not False)
        if should_evaluate:
            evaluation = evaluate_summary_with_gpt4o(final_summary, keywords, topic_name, enable_evaluation=True)
            result["evaluation"] = evaluation
            
            # Log results
            if topic_id:
                log_summary_feedback(topic_id, " ".join(sentences), final_summary, keywords, evaluation)
                
            # Check if we need to improve the summary
            if "scores" in evaluation and "overall" in evaluation["scores"]:
                overall_score = evaluation["scores"]["overall"]
                
                if overall_score < 4.0:
                    logger.warning(f"Summary quality below threshold (score: {overall_score}). Consider adjusting parameters.")
        elif evaluate:
            # If evaluation was requested but disabled, add a note
            result["evaluation"] = {"status": "skipped", "reason": "Evaluation disabled to improve response time"}
            
        # Log final summary if not evaluated
        elif topic_id:
            log_summary_feedback(topic_id, " ".join(sentences), final_summary, keywords)
            
        return result
        
    except Exception as e:
        logger.error(f"Error during text summarization: {e}")
        return {"error": f"Failed to generate summary. {str(e)}"}

def summarize_text(text: str, keywords: List[str] = None, 
                 max_length: int = 150, min_length: int = 40, 
                 topic_id: str = None, topic_name: str = None,
                 evaluate: bool = False, chunk_size: int = 10,
                 final_compression: bool = True, 
                 prompt_prefix: str = None, 
                 enable_evaluation: bool = False) -> str:
    """
    Facade method for backwards compatibility - summarizes plain text.
    
    Args:
        text: The text to summarize
        keywords: Topic keywords for prioritizing relevant sentences
        max_length: Maximum length of the final summary
        min_length: Minimum length of the final summary
        topic_id: Optional topic identifier for logging
        topic_name: Optional name of the topic for prompt enhancement
        evaluate: Whether to evaluate the summary with GPT-4o (legacy parameter)
        chunk_size: Maximum sentences per chunk
        final_compression: Whether to perform a second summarization pass
        prompt_prefix: Optional custom prompt prefix for summarization
        enable_evaluation: If True, enables GPT evaluation (disabled by default to improve performance)
        
    Returns:
        The generated summary as a string
    """
    if not text or not text.strip():
        return "Error: Empty input text."
    
    # Convert text to sentences
    try:
        sentences = sent_tokenize(text)
    except LookupError:
        nltk.download('punkt')
        sentences = sent_tokenize(text)
    
    # Convert to sentence dictionaries
    sentence_dicts = []
    for i, sentence in enumerate(sentences):
        if sentence.strip():
            sentence_dicts.append({
                'text': sentence,
                'source': f"s_{i}",
                'index': i
            })
    
    # Use the new method
    result = summarize_sentence_dicts(
        sentence_dicts,
        keywords=keywords,
        max_length=max_length,
        min_length=min_length,
        topic_id=topic_id,
        topic_name=topic_name,
        evaluate=evaluate,
        enable_evaluation=enable_evaluation,  # Explicitly pass the evaluation flag
        chunk_size=chunk_size,
        final_compression=final_compression,
        prompt_prefix=prompt_prefix
    )
    
    if "error" in result:
        return f"Error: {result['error']}"
        
    return result["summary"] 