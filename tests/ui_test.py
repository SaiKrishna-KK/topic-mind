#!/usr/bin/env python3
"""
TopicMind UI Testing Script
---------------------------
Tests the Streamlit UI of the TopicMind application to verify cross-platform compatibility.
Requires: selenium, webdriver_manager

Usage: 
    pip install selenium webdriver_manager
    python tests/ui_test.py
"""

import os
import time
import logging
from typing import List, Dict, Any, Optional, Union
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ui_tester")

def parse_args():
    parser = argparse.ArgumentParser(description="Test TopicMind Streamlit UI")
    parser.add_argument("--headless", action="store_true", help="Run browser in headless mode")
    parser.add_argument("--host", default="localhost", help="Hostname for the Streamlit app")
    parser.add_argument("--port", default="8501", help="Port for the Streamlit app")
    parser.add_argument("--timeout", type=int, default=60, help="Timeout in seconds for UI operations")
    return parser.parse_args()

def setup_webdriver(headless: bool = False):
    """Set up the WebDriver for browser automation."""
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.service import Service
        from selenium.webdriver.chrome.options import Options
        from webdriver_manager.chrome import ChromeDriverManager
        
        options = Options()
        if headless:
            options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        
        # Set up Chrome WebDriver
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)
        driver.set_page_load_timeout(30)
        logger.info("✅ WebDriver successfully initialized")
        return driver
    except ImportError:
        logger.error("❌ Selenium or webdriver_manager not installed. Run: pip install selenium webdriver_manager")
        return None
    except Exception as e:
        logger.error(f"❌ Failed to initialize WebDriver: {str(e)}")
        return None

def test_streamlit_ui(driver, host: str, port: str, timeout: int):
    """Test the Streamlit UI functionality."""
    results = {
        "ui_loads": False,
        "title_correct": False,
        "input_form_exists": False
    }
    
    try:
        # Open the Streamlit app
        url = f"http://{host}:{port}"
        logger.info(f"Loading Streamlit UI at {url}")
        driver.get(url)
        
        # Wait for page to load
        time.sleep(5)  # Give Streamlit time to render
        
        # Check if page loads
        results["ui_loads"] = True
        logger.info("✅ Streamlit UI loaded successfully")
        
        # Take a screenshot
        screenshot_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                     "logs", "tests", "individual_tests", "streamlit_ui.png")
        driver.save_screenshot(screenshot_path)
        logger.info(f"✅ Screenshot saved to {screenshot_path}")
        
        # Check the page title
        if "TopicMind" in driver.title or "Streamlit" in driver.title:
            results["title_correct"] = True
            logger.info(f"✅ Page title is correct: {driver.title}")
        else:
            logger.warning(f"⚠️ Page title may not be correct: {driver.title}")
        
        # Check if input form elements exist
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        
        # Wait for Streamlit to fully load
        time.sleep(10)
        
        # Look for textarea or text input
        try:
            WebDriverWait(driver, timeout).until(
                EC.presence_of_element_located((By.TAG_NAME, "textarea"))
            )
            results["input_form_exists"] = True
            logger.info("✅ Input form exists")
        except:
            try:
                # Try finding by CSS selector for Streamlit input
                WebDriverWait(driver, timeout).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, ".stTextInput input"))
                )
                results["input_form_exists"] = True
                logger.info("✅ Input form exists")
            except Exception as e:
                logger.warning(f"⚠️ Could not find input form: {str(e)}")
        
        # Print summary
        logger.info("\n=== UI Test Results Summary ===")
        for test, passed in results.items():
            status = "✅ PASSED" if passed else "❌ FAILED"
            logger.info(f"{test}: {status}")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Error during UI testing: {str(e)}")
        return results

def main():
    args = parse_args()
    logger.info("Starting TopicMind UI tests")
    
    # Check if Selenium is installed - if not, provide informational message only
    try:
        import selenium
        from webdriver_manager.chrome import ChromeDriverManager
    except ImportError:
        logger.error("""
        ❌ Selenium not installed. This script requires:
          - selenium
          - webdriver_manager
          - Chrome browser
        
        Install with: pip install selenium webdriver_manager
        """)
        return
    
    # Setup WebDriver
    driver = setup_webdriver(args.headless)
    if not driver:
        return
    
    try:
        # Run UI tests
        test_streamlit_ui(driver, args.host, args.port, args.timeout)
    finally:
        # Clean up
        driver.quit()
        logger.info("UI testing completed")

if __name__ == "__main__":
    main() 