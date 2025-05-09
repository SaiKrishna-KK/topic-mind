import streamlit as st
import requests
import json

# Backend API endpoint
# Make sure the Flask app (app.py) is running
BACKEND_URL = "http://127.0.0.1:5001/analyze"
HEALTH_CHECK_URL = "http://127.0.0.1:5001/health"

# --- UI Configuration ---
st.set_page_config(page_title="TopicMind MVP", layout="wide")

st.title("🧠 TopicMind MVP")
st.caption("Extract Key Topics and Summaries from Text")

# --- Backend Health Check ---
def check_backend_health():
    try:
        response = requests.get(HEALTH_CHECK_URL, timeout=5)
        return response.status_code == 200 and response.json().get("status") == "ok"
    except requests.exceptions.RequestException as e:
        print(f"Backend health check failed: {e}") # Log for debugging
        return False

backend_healthy = check_backend_health()
if not backend_healthy:
    st.error("🚨 Backend service is not running or is unreachable. Please start the backend (run `python app.py` in the topicmind directory) and refresh this page.")
    st.stop() # Stop execution if backend is down
else:
    st.success("✅ Backend service is running.")

# --- Input Area ---
st.subheader("Input Text")
input_text = st.text_area("Paste your text content here:", height=250, placeholder="Enter a long piece of text...")

# Configuration options
st.subheader("Analysis Options")
col1, col2 = st.columns(2)

with col1:
    is_reddit_content = st.checkbox("Clean Reddit UI elements", 
                                    help="Enable this if your text contains Reddit posts/comments with UI elements like upvote counts, timestamps, etc.")
    
with col2:
    num_topics = st.slider("Number of topics to extract", min_value=2, max_value=10, value=5, 
                           help="The maximum number of topics to identify. A smaller text might yield fewer topics regardless of this setting.")

analyze_button = st.button("Analyze Text", type="primary")

# --- Analysis and Output ---
if analyze_button and input_text:
    if not input_text.strip():
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing text... This might take a moment depending on the text length and model processing."):
            try:
                # Prepare the request payload
                payload = json.dumps({
                    "text": input_text,
                    "is_reddit_content": is_reddit_content,
                    "num_topics": num_topics
                })
                headers = {'Content-Type': 'application/json'}

                # Send request to backend
                response = requests.post(BACKEND_URL, data=payload, headers=headers, timeout=120) # Increased timeout for potentially long processing

                response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

                # Process the response
                results_data = response.json()
                results = results_data.get("results", [])

                st.subheader("Analysis Results")
                if results:
                    st.success(f"Found {len(results)} topics.")
                    for i, result in enumerate(results):
                        topic_name = result.get('topic', 'Unnamed Topic')
                        keywords = result.get('keywords', [])
                        keyword_text = ", ".join(keywords) if keywords else "No keywords available"
                        
                        with st.expander(f"**Topic {i+1}: {topic_name}**", expanded=True):
                            st.caption(f"Keywords: {keyword_text}")
                            st.write(result.get('summary', 'No summary available.'))
                        st.divider()
                else:
                    st.info("No topics were identified in the provided text.")

            except requests.exceptions.Timeout:
                st.error("Request timed out. The backend might be taking too long to process the text.")
            except requests.exceptions.RequestException as e:
                st.error(f"Failed to connect to the backend or an error occurred: {e}")
                # Try to get more details from the response if available
                try:
                     error_detail = response.json().get("error", "No additional details.")
                     st.error(f"Backend error details: {error_detail}")
                except Exception:
                    pass # Ignore if response isn't JSON or doesn't have 'error'
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
elif analyze_button:
    st.warning("Please paste some text into the input box before analyzing.")

# --- Footer/Instructions ---
st.markdown("---")
st.caption("""
How to use:
1. **Paste text** into the box (e.g., from a Reddit thread, article, or other long-form content)
2. **Configure options**:
   - Enable "Clean Reddit UI elements" if you're pasting Reddit content with upvotes/timestamps/etc.
   - Adjust the number of topics to extract based on your content length
3. **Click 'Analyze Text'** to process and view the results
""")
