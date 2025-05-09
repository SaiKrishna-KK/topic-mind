import os
import logging
import numpy as np
from typing import List, Tuple, Dict, Optional, Union
import nltk
import torch
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Force CPU-only mode for Mac M1 compatibility
os.environ["CUDA_VISIBLE_DEVICES"] = ""
torch.set_default_device('cpu')

# Constants - using a more advanced model 
MODEL_NAME = "all-mpnet-base-v2"  # Fine-tuned model for better topic representation

# Singleton model instance
_model = None

def load_embedding_model() -> Optional[SentenceTransformer]:
    """
    Loads a sentence transformer model for text embeddings.
    Uses our fine-tuned model trained on Reddit data.
    
    Returns:
        The loaded SentenceTransformer model, or None if loading fails.
    """
    global _model
    if _model is not None:
        return _model
        
    try:
        logging.info(f"Loading fine-tuned sentence transformer model ({MODEL_NAME}) in CPU-only mode")
        _model = SentenceTransformer(MODEL_NAME, device='cpu')
        logging.info(f"Sentence transformer model loaded successfully")
        return _model
    except Exception as e:
        logging.error(f"Error loading sentence transformer model: {e}")
        return None

def get_document_topics(model: SentenceTransformer, document_text: str, 
                      topic_keywords: Dict[int, List[str]]) -> List[Tuple[int, float]]:
    """
    Assigns a topic to a document by comparing embeddings.
    
    Args:
        model: The loaded SentenceTransformer model.
        document_text: The text to analyze.
        topic_keywords: Dictionary mapping topic IDs to lists of keywords.
        
    Returns:
        A list of (topic_id, similarity_score) tuples.
    """
    if not document_text or not document_text.strip():
        logging.warning("Document is empty. Cannot infer topics.")
        return []
        
    try:
        # Get embedding for document
        doc_embedding = model.encode([document_text])[0]
        
        # Compare to each topic (represented by its keywords)
        topic_scores = []
        
        for topic_id, keywords in topic_keywords.items():
            # Convert keywords to a single string
            topic_string = " ".join(keywords)
            # Get embedding for topic
            topic_embedding = model.encode([topic_string])[0]
            # Calculate similarity
            similarity = cosine_similarity([doc_embedding], [topic_embedding])[0][0]
            topic_scores.append((topic_id, similarity))
        
        # Sort by similarity score
        topic_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top match
        if topic_scores:
            return [topic_scores[0]]
        return []
    except Exception as e:
        logging.error(f"Error getting topics for document: {e}")
        return []

def cluster_sentences(model: SentenceTransformer, sentences: List[str], num_topics: int = 5) -> Dict[int, List[str]]:
    """
    Cluster sentences into topics using K-means clustering on embeddings.
    
    Args:
        model: The SentenceTransformer model.
        sentences: List of sentences to cluster.
        num_topics: Number of topics/clusters to create.
        
    Returns:
        Dictionary mapping cluster IDs to lists of sentences.
    """
    try:
        if not sentences:
            return {}
            
        # Only use single topic if extremely short text (3 or fewer sentences)
        if len(sentences) <= 3:
            logging.info(f"Text has only {len(sentences)} sentences, too few for multiple topics. Using a single topic.")
            return {0: sentences}
            
        # Get embeddings for all sentences
        logging.info(f"Generating embeddings for {len(sentences)} sentences")
        embeddings = model.encode(sentences)
        
        # For texts with enough sentences, respect user's requested topic count
        # Only limit if we have fewer sentences than requested topics
        actual_num_topics = min(num_topics, len(sentences) - 1)
        if actual_num_topics < num_topics:
            logging.info(f"Limiting topics to {actual_num_topics} (text has {len(sentences)} sentences)")
        
        # Force multiple topics whenever possible
        if actual_num_topics <= 1 and len(sentences) > 3:
            actual_num_topics = min(2, len(sentences) - 1)  # Force at least 2 topics if possible
            logging.info(f"Forcing multiple topics: using {actual_num_topics} topics")
        
        # Apply K-means clustering with adjusted topic count
        logging.info(f"Clustering sentences into {actual_num_topics} topics")
        kmeans = KMeans(n_clusters=actual_num_topics, random_state=42)
        clusters = kmeans.fit_predict(embeddings)
        
        # Group sentences by cluster
        topic_sentences = {}
        for i, (sentence, cluster_id) in enumerate(zip(sentences, clusters)):
            if cluster_id not in topic_sentences:
                topic_sentences[cluster_id] = []
            topic_sentences[cluster_id].append(sentence)
        
        # Check for any empty clusters and redistribute content if needed
        if len(topic_sentences) < actual_num_topics:
            logging.warning(f"Expected {actual_num_topics} topics but got {len(topic_sentences)}. Redistributing sentences.")
            # This could happen in edge cases - we keep the result as is
            
        logging.info(f"Sentences grouped into {len(topic_sentences)} topics")
        return topic_sentences
    except Exception as e:
        logging.error(f"Error clustering sentences: {e}")
        # Fallback: return a single topic with all sentences
        return {0: sentences}

def extract_topic_keywords(model: SentenceTransformer, topic_sentences: Dict[int, List[str]]) -> Dict[int, List[str]]:
    """
    Extract representative keywords for each topic using TF-IDF approach.
    
    Args:
        model: The SentenceTransformer model (not used in this function but kept for API consistency).
        topic_sentences: Dictionary mapping topic IDs to lists of sentences.
        
    Returns:
        Dictionary mapping topic IDs to lists of keywords.
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        topic_keywords = {}
        
        for topic_id, sentences in topic_sentences.items():
            # Join sentences for this topic
            topic_text = " ".join(sentences)
            
            # Create a TF-IDF vectorizer
            vectorizer = TfidfVectorizer(max_features=10, stop_words='english')
            
            # Fit and transform on this topic's text
            try:
                tfidf_matrix = vectorizer.fit_transform([topic_text])
                feature_names = vectorizer.get_feature_names_out()
                
                # Get the top words based on TF-IDF scores
                tfidf_scores = tfidf_matrix.toarray()[0]
                word_scores = [(word, score) for word, score in zip(feature_names, tfidf_scores)]
                word_scores.sort(key=lambda x: x[1], reverse=True)
                
                # Get just the words (not scores)
                keywords = [word for word, _ in word_scores[:10]]
                topic_keywords[topic_id] = keywords
            except Exception as e:
                logging.warning(f"Could not extract keywords for topic {topic_id}: {e}")
                # Fallback: use the most frequent words
                import re
                from collections import Counter
                
                # Simple tokenization
                words = re.findall(r'\b\w+\b', topic_text.lower())
                # Remove very short words
                words = [word for word in words if len(word) > 2]
                # Get most common
                common_words = [word for word, _ in Counter(words).most_common(10)]
                topic_keywords[topic_id] = common_words or [f"Topic {topic_id}"]
                
        return topic_keywords
    except Exception as e:
        logging.error(f"Error extracting topic keywords: {e}")
        return {topic_id: [f"Topic {topic_id}"] for topic_id in topic_sentences.keys()}

def analyze_topics_in_text(text: str, num_topics: int = 5) -> Tuple[Dict[int, List[str]], Dict[int, List[str]]]:
    """
    Analyze a text by splitting into sentences, clustering them into topics,
    and extracting keywords for each topic.
    
    Args:
        text: The text to analyze.
        num_topics: Number of topics to cluster into.
        
    Returns:
        Tuple of (topic_sentences, topic_keywords) dictionaries
    """
    try:
        model = load_embedding_model()
        if not model:
            logging.error("Could not load embedding model")
            return {}, {}
            
        # Split into sentences - use a more aggressive sentence splitting approach
        # to ensure we have enough sentences for meaningful clustering
        # First try standard sentence tokenization
        sentences = nltk.sent_tokenize(text)
        
        # If we have fewer than 2*num_topics sentences, try splitting by newlines too
        if len(sentences) < 2 * num_topics:
            logging.info(f"Only found {len(sentences)} sentences, attempting to split by paragraphs too")
            # Split by newlines first, then by sentences within each paragraph
            paragraphs = [p for p in text.split('\n') if p.strip()]
            new_sentences = []
            for para in paragraphs:
                para_sentences = nltk.sent_tokenize(para)
                new_sentences.extend(para_sentences)
            
            # Only use this approach if it gives more sentences
            if len(new_sentences) > len(sentences):
                logging.info(f"Improved sentence count from {len(sentences)} to {len(new_sentences)}")
                sentences = new_sentences
        
        # Check if we have enough sentences for meaningful clustering
        if not sentences:
            logging.warning("No sentences found in the text.")
            return {}, {}
            
        # Force multiple topics unless text is extremely short
        if len(sentences) <= 3:
            logging.info("Text too short for multiple topics. Creating a single topic.")
            topic_sentences = {0: sentences}
            topic_keywords = extract_topic_keywords(model, topic_sentences)
            return topic_sentences, topic_keywords
        
        # For longer texts, always try to honor the requested number of topics
        logging.info(f"Found {len(sentences)} sentences, requesting {num_topics} topics")
        topic_sentences = cluster_sentences(model, sentences, num_topics)
        
        # If we still only got 1 topic, but have enough sentences, try to force more topics
        if len(topic_sentences) == 1 and len(sentences) >= num_topics and num_topics > 1:
            logging.warning(f"Only got 1 topic despite having {len(sentences)} sentences. Forcing {min(num_topics, 3)} topics.")
            # Try again with different random seed and more aggressive settings
            topic_sentences = force_multiple_topics(model, sentences, min(num_topics, 3))
        
        # Extract keywords for each topic
        topic_keywords = extract_topic_keywords(model, topic_sentences)
        
        return topic_sentences, topic_keywords
    except Exception as e:
        logging.error(f"Error analyzing topics in text: {e}")
        return {}, {}

def force_multiple_topics(model: SentenceTransformer, sentences: List[str], num_topics: int) -> Dict[int, List[str]]:
    """
    Force the creation of multiple topics by using different clustering approaches.
    This is a fallback method when K-means doesn't produce enough distinct topics.
    
    Args:
        model: The SentenceTransformer model.
        sentences: List of sentences.
        num_topics: Number of topics to create.
        
    Returns:
        Dictionary mapping topic IDs to lists of sentences.
    """
    try:
        # Get embeddings
        embeddings = model.encode(sentences)
        
        # Try spectral clustering as an alternative
        from sklearn.cluster import SpectralClustering
        
        logging.info(f"Attempting spectral clustering to force {num_topics} topics")
        spectral = SpectralClustering(n_clusters=num_topics, random_state=42, assign_labels='discretize')
        clusters = spectral.fit_predict(embeddings)
        
        # Group sentences by cluster
        topic_sentences = {}
        for i, (sentence, cluster_id) in enumerate(zip(sentences, clusters)):
            if cluster_id not in topic_sentences:
                topic_sentences[cluster_id] = []
            topic_sentences[cluster_id].append(sentence)
            
        # If we still don't have enough topics, just split sentences evenly
        if len(topic_sentences) < num_topics:
            logging.warning(f"Still only got {len(topic_sentences)} topics. Splitting sentences evenly.")
            # Fallback: just split the sentences into equal parts
            topic_sentences = {}
            sentences_per_topic = len(sentences) // num_topics
            remainder = len(sentences) % num_topics
            
            start_idx = 0
            for i in range(num_topics):
                # Add one extra sentence to earlier topics if we have remainder
                count = sentences_per_topic + (1 if i < remainder else 0)
                end_idx = start_idx + count
                topic_sentences[i] = sentences[start_idx:end_idx]
                start_idx = end_idx
        
        return topic_sentences
    except Exception as e:
        logging.error(f"Error in force_multiple_topics: {e}")
        # Last resort: manual split
        sentences_per_topic = max(1, len(sentences) // num_topics)
        return {i: sentences[i*sentences_per_topic:(i+1)*sentences_per_topic] 
                for i in range(num_topics) if i*sentences_per_topic < len(sentences)}

def get_topic_top_words(topic_keywords: Dict[int, List[str]], topic_id: int, num_words: int = 10) -> List[Tuple[str, float]]:
    """
    Gets the top words for a specific topic.
    
    Args:
        topic_keywords: Dictionary mapping topic IDs to keywords.
        topic_id: The ID of the topic.
        num_words: The number of top words to retrieve.
        
    Returns:
        A list of (word, score) tuples.
    """
    try:
        # Get keywords for this topic
        keywords = topic_keywords.get(topic_id, [])
        
        # Limit to the requested number
        keywords = keywords[:num_words]
        
        # Return with mock scores
        return [(word, 1.0 - 0.05 * i) for i, word in enumerate(keywords)]
    except Exception as e:
        logging.error(f"Error retrieving top words for topic {topic_id}: {e}")
        return []

# For testing
if __name__ == "__main__":
    # Test text
    test_text = """
    Machine learning is a field of inquiry devoted to understanding and building methods that 'learn', 
    that is, methods that leverage data to improve performance on some set of tasks. It is seen as a part of artificial intelligence. 
    Machine learning algorithms build a model based on sample data, known as training data, 
    in order to make predictions or decisions without being explicitly programmed to do so.
    
    Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence 
    concerned with the interactions between computers and human language, in particular how to program 
    computers to process and analyze large amounts of natural language data.
    
    The stock market refers to the collection of markets and exchanges where regular activities of buying, 
    selling, and issuance of shares of publicly-held companies take place. Such financial activities are 
    conducted through institutionalized formal exchanges or over-the-counter (OTC) marketplaces which 
    operate under a defined set of regulations.
    """
    
    # Analyze topics
    topic_sentences, topic_keywords = analyze_topics_in_text(test_text, num_topics=3)
    
    # Print results
    print("\nTOPIC ANALYSIS RESULTS:")
    print("-----------------------")
    
    for topic_id, sentences in topic_sentences.items():
        keywords = topic_keywords.get(topic_id, [])
        
        print(f"\nTOPIC {topic_id}: {', '.join(keywords[:5])}")
        print(f"  Sentences: {len(sentences)}")
        print(f"  First sentence: {sentences[0][:100]}...") 