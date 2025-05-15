#!/usr/bin/env python3
import os
import json
import argparse
import logging
from datetime import datetime
from typing import List, Dict, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"logs/test_summarizer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Ensure we track changes
os.makedirs("logs", exist_ok=True)
CHANGE_TRACKER_PATH = "logs/change_tracker.md"

def log_change(message: str):
    """Add an entry to the change tracker log"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Create file if it doesn't exist
    if not os.path.exists(CHANGE_TRACKER_PATH):
        with open(CHANGE_TRACKER_PATH, 'w') as f:
            f.write("# Summarizer Change Tracker\n\n")
    
    # Append the new change
    with open(CHANGE_TRACKER_PATH, 'a') as f:
        f.write(f"## {timestamp}\n\n{message}\n\n")
    
    logger.info(f"Change logged: {message}")

def load_sample_data(filepath: str) -> List[Dict]:
    """Load sample data from a JSON file"""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        logger.info(f"Loaded sample data from {filepath}: {len(data)} items")
        return data
    except Exception as e:
        logger.error(f"Error loading sample data: {e}")
        return []

def create_dummy_data() -> List[Dict]:
    """Create dummy data with sample text and keywords"""
    dummy_data = [
        {
            "id": "dummy1",
            "text": """
            Machine learning is transforming how we interact with technology. 
            Deep learning models have shown remarkable ability to understand natural language. 
            However, many users express concerns about privacy implications. 
            Data collection practices need more transparency according to critics.
            Some researchers argue we need stronger regulations on AI.
            The benefits of AI are vast, from healthcare to climate science.
            Neural networks can now recognize patterns humans might miss.
            Training large models requires significant computational resources.
            Open source communities have contributed to making AI more accessible.
            Ethical considerations should be central to AI development.
            """,
            "keywords": ["machine learning", "ai", "deep learning", "privacy", "ethics"]
        },
        {
            "id": "dummy2",
            "text": """
            Reddit discussions on gaming often focus on new releases.
            Players frequently debate the merits of different consoles.
            PC gaming has a dedicated community that values customization.
            Many threads address issues of representation in games.
            Some users share tutorials and guides for difficult game sections.
            Game developers occasionally participate in AMA (Ask Me Anything) sessions.
            Nostalgia for classic games is a common theme in gaming subreddits.
            Competitive gaming and esports have growing discussion communities.
            Mobile gaming is sometimes viewed with skepticism by traditional gamers.
            Indie games receive significant attention and support from Reddit users.
            """,
            "keywords": ["gaming", "reddit", "discussion", "community", "players"]
        }
    ]
    
    logger.info(f"Created {len(dummy_data)} dummy data items")
    return dummy_data

def run_test(sample_file: Optional[str] = None, use_openai: bool = False):
    """Run the summarizer test with provided sample or dummy data"""
    # Import the summarizer module
    try:
        from models.bart_summarizer import load_summarizer_model, summarize_text
        logger.info("Imported summarizer module successfully")
    except ImportError as e:
        logger.error(f"Error importing summarizer module: {e}")
        return
    
    # Load the model
    logger.info("Loading summarizer model...")
    success, message = load_summarizer_model()
    if not success:
        logger.error(f"Failed to load summarizer model: {message}")
        return
    
    # Load or create test data
    if sample_file and os.path.exists(sample_file):
        test_data = load_sample_data(sample_file)
    else:
        logger.info("No valid sample file provided, using dummy data")
        test_data = create_dummy_data()
    
    if not test_data:
        logger.error("No test data available")
        return
    
    # Process each item
    for i, item in enumerate(test_data):
        topic_id = item.get("id", f"topic_{i}")
        text = item.get("text", "").strip()
        keywords = item.get("keywords", [])
        
        if not text:
            logger.warning(f"Empty text for item {topic_id}, skipping")
            continue
        
        logger.info(f"Processing item {topic_id} with {len(text)} chars")
        logger.info(f"Keywords: {keywords}")
        
        # Generate summary
        summary = summarize_text(
            text=text,
            keywords=keywords,
            topic_id=topic_id,
            evaluate=use_openai
        )
        
        print(f"\n--- Summary for {topic_id} ---")
        print(f"Keywords: {', '.join(keywords)}")
        print(f"\n{summary}\n")
        print("----------------------------\n")
        
        # If not using OpenAI for evaluation, manually check quality
        if not use_openai:
            log_change(f"Generated summary for topic {topic_id}:\n\n```\n{summary}\n```\n\nKeywords: {keywords}")
    
    logger.info("Test completed successfully")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the Reddit summarizer functionality")
    parser.add_argument("--sample", help="Path to sample data JSON file", default=None)
    parser.add_argument("--openai", action="store_true", help="Use OpenAI for evaluation")
    
    args = parser.parse_args()
    
    # Check if using OpenAI but no API key is set
    if args.openai and not os.environ.get("OPENAI_API_KEY"):
        logger.warning("OpenAI evaluation requested but OPENAI_API_KEY not set in environment")
        print("Warning: OpenAI API key not found. To use evaluation, set the OPENAI_API_KEY environment variable.")
        print("Continuing without OpenAI evaluation...")
        args.openai = False
    
    log_change("Starting test run with the new TF-IDF + DistilBART summarizer")
    run_test(args.sample, args.openai) 