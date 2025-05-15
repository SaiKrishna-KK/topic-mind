#!/usr/bin/env python3
"""
TopicMind API Testing Script
----------------------------
Tests the API endpoints of the TopicMind application in a cross-platform manner.
"""

import requests
import json
import time
import os
import logging
from typing import Dict, Any, Optional, List, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("api_tester")

# API Configuration
API_HOST = os.environ.get('API_HOST', 'localhost')
API_PORT = os.environ.get('API_PORT', '5001')
API_BASE_URL = f"http://{API_HOST}:{API_PORT}"

# Test data
TEST_TEXT = """
Machine learning is a field of study in artificial intelligence concerned with the development
and study of statistical algorithms that can learn from data and generalize to unseen data,
and thus perform tasks without explicit instructions.
"""

def check_api_health() -> bool:
    """Check if the API is healthy."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            logger.info("✅ API health check passed")
            return True
        else:
            logger.error(f"❌ API health check failed with status code {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ API health check failed with error: {str(e)}")
        return False

def test_topic_analysis(text: str) -> Union[Dict[str, Any], None]:
    """Test the topic analysis endpoint."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/analyze_topics",
            json={"text": text, "num_topics": 2},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            logger.info(f"✅ Topic analysis successful. Found {len(result.get('topics', []))} topics")
            return result
        else:
            logger.error(f"❌ Topic analysis failed with status code {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Topic analysis failed with error: {str(e)}")
        return None

def test_summarization(text: str) -> Union[Dict[str, Any], None]:
    """Test the summarization endpoint."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/summarize",
            json={"text": text, "max_length": 100},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            logger.info(f"✅ Summarization successful: {result.get('summary', '')[:50]}...")
            return result
        else:
            logger.error(f"❌ Summarization failed with status code {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Summarization failed with error: {str(e)}")
        return None

def run_all_tests() -> Dict[str, bool]:
    """Run all API tests and return results."""
    results = {
        "health": False,
        "topic_analysis": False,
        "summarization": False
    }
    
    # Health check
    results["health"] = check_api_health()
    
    if not results["health"]:
        logger.error("Skipping further tests due to failed health check")
        return results
    
    # Topic analysis
    topic_result = test_topic_analysis(TEST_TEXT)
    results["topic_analysis"] = topic_result is not None
    
    # Summarization
    summary_result = test_summarization(TEST_TEXT)
    results["summarization"] = summary_result is not None
    
    # Print summary
    logger.info("\n=== Test Results Summary ===")
    for test, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        logger.info(f"{test}: {status}")
    
    return results

if __name__ == "__main__":
    logger.info("Starting TopicMind API tests")
    run_all_tests()
    logger.info("API testing completed") 