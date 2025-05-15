#!/usr/bin/env python3
import os
import json
import logging
import argparse
from datetime import datetime
from typing import List, Dict, Any
from dotenv import load_dotenv
import pprint

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"logs/test_thread_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)
os.makedirs("logs/pipeline", exist_ok=True)

def load_test_data(file_path: str) -> str:
    """Load text from a test file"""
    try:
        with open(file_path, 'r') as f:
            return f.read()
    except Exception as e:
        logger.error(f"Error loading test data: {e}")
        return ""

def get_topic_name_from_bertopic(topic_id: int, topic_model=None, keywords: List[str] = None) -> str:
    """
    Get topic name from BERTopic model or generate from keywords as fallback.
    
    Args:
        topic_id: The ID of the topic in the BERTopic model
        topic_model: Optional BERTopic model instance
        keywords: Optional list of keywords to use as fallback
        
    Returns:
        A string representing the topic name
    """
    # Try to get topic name from BERTopic model
    if topic_model:
        try:
            # BERTopic provides topic words with their weights
            topic_words = topic_model.get_topic(topic_id)
            if topic_words:
                # Return the top word as a simple topic name
                return topic_words[0][0]
        except Exception as e:
            logger.warning(f"Failed to get topic name from BERTopic: {e}")
    
    # Fallback to keywords
    if keywords and len(keywords) > 0:
        # Use first 2-3 keywords to form a topic name
        top_keywords = keywords[:min(3, len(keywords))]
        topic_name = " ".join(top_keywords)
        return topic_name
    
    # If all else fails
    return f"Topic {topic_id}"

def test_thread_refiner(text: str, keywords: List[str], max_sentences: int = 15) -> List[Dict[str, Any]]:
    """Test the thread refiner component"""
    try:
        from utils.thread_refiner import refine_thread_content
        
        # For Reddit content, clean it first
        from utils.preprocessor import clean_reddit_content
        cleaned_text = clean_reddit_content(text)
        logger.info(f"Cleaned Reddit content: {len(text)} chars -> {len(cleaned_text)} chars")
        
        # Process with thread refiner
        logger.info(f"Refining content with keywords: {keywords}")
        refined_sentences = refine_thread_content(
            cleaned_text, 
            topic_keywords=keywords,
            max_sentences=max_sentences
        )
        
        logger.info(f"Thread refiner produced {len(refined_sentences)} refined sentences")
        return refined_sentences
    except Exception as e:
        logger.error(f"Error in thread refiner: {e}")
        return []

def test_chunked_summarization(sentence_dicts: List[Dict[str, Any]], 
                              keywords: List[str],
                              topic_name: str = None,
                              chunk_size: int = 10,
                              final_compression: bool = True,
                              prompt_prefix: str = None,
                              topic_id: int = 0) -> Dict[str, Any]:
    """Test the improved chunked summarization component with two-pass approach"""
    try:
        from models.bart_summarizer import load_summarizer_model, summarize_sentence_dicts
        
        # Load the model
        success, message = load_summarizer_model()
        if not success:
            logger.error(f"Failed to load summarizer model: {message}")
            return {"error": message}
        
        # If no topic name, try to generate one using BERTopic fallback
        if not topic_name:
            try:
                # Try to import BERTopic model if available
                from models.bertopic_model import load_topic_model
                topic_model = load_topic_model()
                if topic_model:
                    topic_name = get_topic_name_from_bertopic(topic_id, topic_model, keywords)
                    logger.info(f"Generated topic name from BERTopic: {topic_name}")
                else:
                    # Fallback to keywords
                    topic_name = get_topic_name_from_bertopic(topic_id, None, keywords)
                    logger.info(f"Generated topic name from keywords: {topic_name}")
            except Exception as e:
                logger.warning(f"Error loading BERTopic model: {e}")
                # Fallback to keywords
                topic_name = get_topic_name_from_bertopic(topic_id, None, keywords)
                logger.info(f"Fallback to keyword-based topic name: {topic_name}")
        
        # Summarize with chunking and two-pass approach
        logger.info(f"Summarizing {len(sentence_dicts)} sentences with chunk_size={chunk_size}")
        logger.info(f"Using final compression: {final_compression}, Topic name: {topic_name}")
        
        summary_result = summarize_sentence_dicts(
            sentence_dicts=sentence_dicts,
            keywords=keywords,
            topic_name=topic_name,
            chunk_size=chunk_size,
            final_compression=final_compression,
            prompt_prefix=prompt_prefix,
            evaluate=True,  # Use GPT-4o evaluation if API key is available
            topic_id=str(topic_id)
        )
        
        if "error" in summary_result:
            logger.error(f"Summarization error: {summary_result['error']}")
        else:
            logger.info(f"Generated summary with {len(summary_result['summary'])} chars")
            logger.info(f"Chunked summarization: {summary_result['chunked']}")
            logger.info(f"Final compression applied: {summary_result.get('final_compressed', False)}")
            
        return summary_result
    except Exception as e:
        logger.error(f"Error in chunked summarization: {e}")
        return {"error": str(e)}

def format_evaluation_scores(evaluation: Dict[str, Any]) -> str:
    """Format evaluation scores into a readable string"""
    if not evaluation or "scores" not in evaluation:
        return "No evaluation scores available"
    
    scores = evaluation["scores"]
    result = []
    
    for dimension, score in scores.items():
        if score is not None:
            stars = "★" * int(score) + "☆" * (5 - int(score))
            result.append(f"{dimension.capitalize()}: {score}/5 {stars}")
    
    if "overall" in scores and scores["overall"] is not None:
        overall = scores["overall"]
        overall_stars = "★" * int(overall) + "☆" * (5 - int(overall))
        result.append(f"\nOverall: {overall:.1f}/5 {overall_stars}")
    
    return "\n".join(result)

def run_full_pipeline_test(input_file: str, keywords: List[str], 
                         topic_name: str = None,
                         max_sentences: int = 15, 
                         chunk_size: int = 10,
                         final_compression: bool = True,
                         prompt_prefix: str = None,
                         topic_id: int = 0) -> None:
    """Run the full pipeline test from refinement to summarization with enhanced logging"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Print header
    print("\n" + "="*80)
    print("TOPICMIND ENHANCED THREAD SUMMARIZATION PIPELINE TEST")
    print("="*80 + "\n")
    
    # Load the test data
    text = load_test_data(input_file)
    if not text:
        print(f"Error: Failed to load test data from {input_file}")
        return
    
    print(f"Loaded test data: {len(text)} characters from {input_file}")
    
    # If topic name not provided, use BERTopic or keywords as fallback
    if not topic_name:
        try:
            # Try to load BERTopic model
            try:
                from models.bertopic_model import load_topic_model
                topic_model = load_topic_model()
                if topic_model:
                    topic_name = get_topic_name_from_bertopic(topic_id, topic_model, keywords)
            except Exception:
                pass
            
            # If still no topic name, fallback to keywords
            if not topic_name:
                topic_name = get_topic_name_from_bertopic(topic_id, None, keywords)
                
            print(f"Auto-generated topic name: {topic_name}")
        except Exception as e:
            logger.warning(f"Failed to generate topic name: {e}")
            topic_name = None
    
    print(f"Topic name: {topic_name or 'Not specified'}")
    print(f"Topic keywords: {', '.join(keywords)}")
    print(f"Max sentences: {max_sentences}")
    print(f"Chunk size: {chunk_size}")
    print(f"Final compression: {final_compression}")
    if prompt_prefix:
        print(f"Custom prompt: {prompt_prefix}")
    else:
        print("Using auto-generated prompts")
    print("\n" + "-"*80 + "\n")
    
    # Step 1: Thread refinement
    print("STEP 1: Thread Content Refinement")
    print("-"*30)
    refined_sentences = test_thread_refiner(text, keywords, max_sentences)
    
    if not refined_sentences:
        print("Error: Thread refinement failed to produce any sentences.")
        return
    
    print(f"✅ Successfully refined to {len(refined_sentences)} sentences with provenance tracking")
    print("\nSample sentences:")
    for i, sentence in enumerate(refined_sentences[:3]):
        print(f"{i+1}. {sentence['text']} (ID: {sentence['source']})")
    
    if len(refined_sentences) > 3:
        print(f"... plus {len(refined_sentences)-3} more sentences")
    
    print("\n" + "-"*80 + "\n")
    
    # Step 2: Enhanced summarization with two-pass approach
    print("STEP 2: Enhanced Two-Pass Summarization")
    print("-"*30)
    summary_result = test_chunked_summarization(
        refined_sentences, 
        keywords, 
        topic_name=topic_name,
        chunk_size=chunk_size,
        final_compression=final_compression,
        prompt_prefix=prompt_prefix,
        topic_id=topic_id
    )
    
    if "error" in summary_result:
        print(f"Error: Summarization failed: {summary_result['error']}")
        return
    
    print("✅ Successfully generated summary")
    
    # Print prompts used
    if "chunk_prompt" in summary_result:
        print(f"\nChunk-level prompt: {summary_result.get('chunk_prompt', 'Not available')}")
    if "final_prompt" in summary_result:
        print(f"Final summary prompt: {summary_result.get('final_prompt', 'Not available')}")
    
    # Print chunk summaries
    if "chunk_summaries" in summary_result and summary_result["chunk_summaries"]:
        print("\nCHUNK SUMMARIES:")
        print("-"*30)
        for i, chunk_summary in enumerate(summary_result["chunk_summaries"]):
            print(f"Chunk {i+1}:")
            print(f"{chunk_summary}\n")
    
    # Print final summary
    print("\nFINAL SUMMARY:")
    print("-"*30)
    print(summary_result["summary"])
    
    print(f"\nContext-aware chunking: {'Yes' if summary_result.get('chunked', False) else 'No'}")
    print(f"Final compression applied: {'Yes' if summary_result.get('final_compressed', False) else 'No'}")
    print(f"Summary length: {len(summary_result['summary'])} characters")
    
    # Display evaluation results if available
    if "evaluation" in summary_result:
        evaluation = summary_result["evaluation"]
        print("\nGPT-4o EVALUATION:")
        print("-"*30)
        
        if "error" in evaluation:
            print(f"Evaluation error: {evaluation['error']}")
        else:
            print(format_evaluation_scores(evaluation))
            
            # Print full feedback if available
            if "feedback" in evaluation:
                print("\nFEEDBACK DETAILS:")
                print("-"*20)
                print(evaluation["feedback"])
    
    # Save full results for inspection with timestamp
    result_path = f"logs/pipeline/enhanced_pipeline_result_{timestamp}.json"
    with open(result_path, 'w') as f:
        # Create a clean version for JSON serialization
        clean_result = {
            "timestamp": timestamp,
            "input_file": input_file,
            "topic_name": topic_name,
            "auto_generated_topic_name": topic_name if not prompt_prefix else None,
            "keywords": keywords,
            "max_sentences": max_sentences,
            "chunk_size": chunk_size,
            "final_compression": final_compression,
            "custom_prompt": prompt_prefix,
            "chunk_prompt": summary_result.get("chunk_prompt", None),
            "final_prompt": summary_result.get("final_prompt", None),
            "refined_sentences_count": len(refined_sentences),
            "chunk_count": len(summary_result.get("chunk_summaries", [])),
            "chunk_summaries": summary_result.get("chunk_summaries", []),
            "final_summary": summary_result["summary"],
            "evaluation": summary_result.get("evaluation", {})
        }
        json.dump(clean_result, f, indent=2)
    
    # Save a more readable summary report
    report_path = f"logs/pipeline/enhanced_pipeline_report_{timestamp}.txt"
    with open(report_path, 'w') as f:
        f.write("TOPICMIND ENHANCED SUMMARIZATION PIPELINE REPORT\n")
        f.write("="*60 + "\n\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Input file: {input_file}\n")
        f.write(f"Topic name: {topic_name or 'Auto-generated'}\n")
        f.write(f"Keywords: {', '.join(keywords)}\n")
        f.write(f"Max sentences: {max_sentences}\n")
        f.write(f"Chunk size: {chunk_size}\n")
        f.write(f"Final compression: {final_compression}\n\n")
        
        f.write("PROMPTS USED\n")
        f.write("-"*30 + "\n")
        f.write(f"Custom prompt: {prompt_prefix or 'None'}\n")
        f.write(f"Chunk prompt: {summary_result.get('chunk_prompt', 'Not available')}\n")
        f.write(f"Final prompt: {summary_result.get('final_prompt', 'Not available')}\n\n")
        
        f.write("REFINED SENTENCES\n")
        f.write("-"*30 + "\n")
        f.write(f"Total sentences: {len(refined_sentences)}\n\n")
        
        f.write("CHUNK SUMMARIES\n")
        f.write("-"*30 + "\n")
        for i, chunk_summary in enumerate(summary_result.get("chunk_summaries", [])):
            f.write(f"Chunk {i+1}:\n{chunk_summary}\n\n")
        
        f.write("FINAL SUMMARY\n")
        f.write("-"*30 + "\n")
        f.write(summary_result["summary"] + "\n\n")
        
        if "evaluation" in summary_result and "scores" in summary_result["evaluation"]:
            f.write("EVALUATION SCORES\n")
            f.write("-"*30 + "\n")
            f.write(format_evaluation_scores(summary_result["evaluation"]) + "\n\n")
            
            if "feedback" in summary_result["evaluation"]:
                f.write("FEEDBACK DETAILS\n")
                f.write("-"*30 + "\n")
                f.write(summary_result["evaluation"]["feedback"] + "\n")
    
    print(f"\nFull test results saved to: {result_path}")
    print(f"Readable report saved to: {report_path}")
    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80 + "\n")

def main():
    """Main function to parse arguments and run the test"""
    parser = argparse.ArgumentParser(description="Test the TopicMind enhanced summarization pipeline")
    parser.add_argument("--input", help="Path to input text file", default="reddit_thread.txt")
    parser.add_argument("--keywords", help="Comma-separated list of topic keywords", 
                      default="sepsis,infection,emergency,urgent care,symptoms")
    parser.add_argument("--topic-name", help="Topic name for prompt customization", 
                      default=None)  # Changed to None to test auto-generation
    parser.add_argument("--max-sentences", type=int, default=15, help="Maximum sentences to extract")
    parser.add_argument("--chunk-size", type=int, default=10, help="Chunk size for summarization")
    parser.add_argument("--no-compression", action="store_true", help="Disable final compression (second pass)")
    parser.add_argument("--prompt", help="Custom prompt prefix for summarization")
    parser.add_argument("--topic-id", type=int, default=0, help="Topic ID for BERTopic model lookup")
    parser.add_argument("--batch-test", action="store_true", help="Run tests on multiple files")
    
    args = parser.parse_args()
    
    # Convert keywords string to list
    keywords = [k.strip() for k in args.keywords.split(",")]
    
    # Update log in development pipeline progress
    with open("logs/dev_pipeline_progress.md", "a") as f:
        f.write(f"\n\n## {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write(f"### Enhanced Pipeline Test Run\n\n")
        f.write(f"- Input file: `{args.input}`\n")
        f.write(f"- Topic name: {args.topic_name or 'Auto-generated'}\n")
        f.write(f"- Keywords: {keywords}\n")
        f.write(f"- Max sentences: {args.max_sentences}\n")
        f.write(f"- Chunk size: {args.chunk_size}\n")
        f.write(f"- Final compression: {not args.no_compression}\n")
        if args.prompt:
            f.write(f"- Custom prompt: {args.prompt}\n")
    
    # Run batch tests if specified
    if args.batch_test:
        test_files = [
            "reddit_thread.txt",
            "reddit_thread_relationship.txt", 
            "reddit_thread_career.txt", 
            "reddit_thread_gaming.txt"
        ]
        
        for test_file in test_files:
            print(f"\n\nTESTING WITH {test_file}\n{'-'*40}")
            
            # Run with different keyword sets based on the file
            if "relationship" in test_file:
                file_keywords = ["relationship", "dating", "partner", "breakup", "marriage"]
            elif "career" in test_file:
                file_keywords = ["career", "job", "interview", "resume", "workplace"]
            elif "gaming" in test_file: 
                file_keywords = ["game", "gaming", "console", "playstation", "xbox"]
            else:
                file_keywords = keywords
                
            run_full_pipeline_test(
                input_file=test_file,
                keywords=file_keywords,
                topic_name=None,  # Force auto-generation
                max_sentences=args.max_sentences,
                chunk_size=args.chunk_size,
                final_compression=not args.no_compression,
                prompt_prefix=args.prompt,
                topic_id=args.topic_id
            )
    else:
        # Run the full pipeline test on a single file
        run_full_pipeline_test(
            input_file=args.input,
            keywords=keywords,
            topic_name=args.topic_name,
            max_sentences=args.max_sentences,
            chunk_size=args.chunk_size,
            final_compression=not args.no_compression,
            prompt_prefix=args.prompt,
            topic_id=args.topic_id
        )

if __name__ == "__main__":
    main() 