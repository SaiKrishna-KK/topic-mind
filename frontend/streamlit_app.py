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
input_text = st.text_area("Paste your Reddit thread or long text here:", height=250, placeholder="Enter a long piece of text...")

analyze_button = st.button("Analyze Text", type="primary")

# --- Analysis and Output ---
if analyze_button and input_text:
    if not input_text.strip():
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing text... This might take a moment depending on the text length and model processing."):
            try:
                # Prepare the request payload
                payload = json.dumps({"text": input_text})
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
                        with st.expander(f"**Topic {i+1}: {result.get('topic', 'N/A')}**", expanded=True):
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
st.caption("How to use: Paste text (e.g., from a Reddit comment thread) into the box and click 'Analyze Text'.")
