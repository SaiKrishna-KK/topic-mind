#!/usr/bin/env python3
import os
import json
import argparse
import logging
from datetime import datetime
from typing import List, Dict
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"logs/test_semantic_refiner_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

def evaluate_with_gpt4o(sentences: List[str], keywords: List[str]) -> Dict:
    """
    Evaluate the refined sentences using GPT-4o for quality assessment.
    
    Args:
        sentences: The refined sentences to evaluate
        keywords: The topic keywords
        
    Returns:
        Dictionary with evaluation scores and feedback
    """
    try:
        from openai import OpenAI
        
        # Load API key from environment
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OpenAI API key not found. Skipping evaluation.")
            return {"error": "OpenAI API key not found"}
            
        client = OpenAI()
        
        # Prepare the evaluation prompt
        eval_prompt = f"""
        Evaluate the following Reddit topic sentences for clarity, coherence, and relevance to the topic keywords. 
        Rate each dimension from 1 to 5 and explain briefly:
        
        Sentences: {sentences}
        
        Keywords: {keywords}
        """
        
        # Call the API
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": eval_prompt}
            ]
        )
        
        feedback = completion.choices[0].message.content
        
        # Extract scores using regex
        import re
        clarity_match = re.search(r"clarity:?\s*(\d+)", feedback, re.IGNORECASE)
        coherence_match = re.search(r"coherence:?\s*(\d+)", feedback, re.IGNORECASE)
        relevance_match = re.search(r"relevance:?\s*(\d+)", feedback, re.IGNORECASE)
        
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
            
        # Log the evaluation to a file
        with open("logs/gpt_validation_log.json", "w") as f:
            json.dump(result, f, indent=2)
            
        return result
        
    except Exception as e:
        logger.error(f"Error during GPT evaluation: {e}")
        return {"error": str(e)}

def main():
    parser = argparse.ArgumentParser(description="Test the Reddit semantic refiner functionality")
    parser.add_argument("--input", help="Path to input Reddit text file", default="reddit_thread.txt")
    parser.add_argument("--max-sentences", type=int, default=15, help="Maximum number of sentences to return")
    parser.add_argument("--no-eval", action="store_true", help="Skip GPT-4o evaluation")
    
    args = parser.parse_args()
    
    # Print header
    print("\n" + "="*70)
    print("REDDIT SEMANTIC REFINER TEST")
    print("="*70 + "\n")
    
    # Check if the input file exists
    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        print(f"Error: Input file not found: {args.input}")
        return
    
    # Import the semantic refiner
    try:
        from utils.reddit_semantic_refiner import process_reddit_text
        logger.info("Imported semantic refiner module successfully")
    except ImportError as e:
        logger.error(f"Error importing semantic refiner module: {e}")
        print(f"Error: Failed to import semantic refiner module: {e}")
        return
    
    # Set up test keywords
    keywords = ['sepsis', 'infection', 'emergency', 'urgent care', 'symptoms']
    logger.info(f"Using test keywords: {keywords}")
    
    # Load the Reddit text
    try:
        with open(args.input, 'r') as f:
            reddit_text = f.read()
        logger.info(f"Loaded Reddit text from {args.input}: {len(reddit_text)} chars")
    except Exception as e:
        logger.error(f"Error loading Reddit text: {e}")
        print(f"Error: Failed to load Reddit text: {e}")
        return
    
    # Process the text
    logger.info(f"Processing Reddit text with max_sentences={args.max_sentences}")
    print(f"Processing Reddit text from {args.input}...")
    print(f"Keywords: {', '.join(keywords)}")
    print(f"Max sentences: {args.max_sentences}")
    
    try:
        refined_sentences = process_reddit_text(
            raw_text=reddit_text, 
            topic_keywords=keywords,
            max_sentences=args.max_sentences
        )
        
        if not refined_sentences:
            logger.warning("No refined sentences produced")
            print("Warning: No refined sentences were produced.")
            return
            
        logger.info(f"Successfully refined to {len(refined_sentences)} sentences")
        
        # Print the results
        print("\n" + "-"*70)
        print(f"TOP {len(refined_sentences)} REFINED SENTENCES:")
        print("-"*70)
        
        for i, sentence in enumerate(refined_sentences, 1):
            print(f"{i}. {sentence}")
            
        # Evaluate with GPT-4o if not disabled
        if not args.no_eval:
            # Check if OpenAI API key is set
            if not os.environ.get("OPENAI_API_KEY"):
                logger.warning("OpenAI API key not set. Skipping evaluation.")
                print("\nWarning: OpenAI API key not set. Skipping GPT-4o evaluation.")
                print("To enable evaluation, set the OPENAI_API_KEY environment variable.")
                return
                
            print("\n" + "-"*70)
            print("EVALUATING WITH GPT-4o...")
            print("-"*70)
            
            evaluation = evaluate_with_gpt4o(refined_sentences, keywords)
            
            if "error" in evaluation:
                logger.error(f"Evaluation error: {evaluation['error']}")
                print(f"\nError during evaluation: {evaluation['error']}")
                return
                
            print("\nGPT-4o EVALUATION:")
            print(f"\n{evaluation['feedback']}")
            
            # Print scores
            if "scores" in evaluation:
                scores = evaluation["scores"]
                print("\nSCORES:")
                for dimension, score in scores.items():
                    if score is not None:
                        print(f"- {dimension.capitalize()}: {score}/5")
                        
                # Check if any score is below threshold
                if any(score is not None and score < 4 for score in scores.values() if score is not None):
                    print("\nACTION REQUIRED: At least one score is below 4/5.")
                    print("Consider tuning the filtering, deduplication threshold, or TF-IDF weights.")
                else:
                    print("\nSUCCESS: All scores are 4/5 or higher!")
        
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        print(f"Error: Processing failed: {e}")
        
    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70 + "\n")

if __name__ == "__main__":
    main() 