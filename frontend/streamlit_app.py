import streamlit as st
import requests
import json
import datetime
import os

# Backend API endpoint
# Make sure the Flask app (app.py) is running
BACKEND_URL = "http://127.0.0.1:5001/analyze"
HEALTH_CHECK_URL = "http://127.0.0.1:5001/health"

# --- UI Configuration ---
st.set_page_config(
    page_title="TopicMind - Smart Text Summarization", 
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/topicmind',
        'Report a bug': 'https://github.com/yourusername/topicmind/issues',
        'About': 'TopicMind: A smart text summarization tool for extracting insights from discussions.'
    }
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem !important;
        color: #1E88E5;
    }
    .subheader {
        font-size: 1.5rem;
        margin-bottom: 1rem;
    }
    .keyword-tag {
        background-color: #f0f2f6;
        padding: 5px 10px;
        border-radius: 15px;
        margin: 2px;
        display: inline-block;
    }
    .summary-box {
        background-color: #f7f7f7;
        padding: 15px;
        border-radius: 5px;
        border-left: 3px solid #1E88E5;
    }
    .info-text {
        color: #555;
        font-size: 0.9rem;
    }
    .divider {
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">TopicMind</h1>', unsafe_allow_html=True)
st.markdown('<p class="info-text">Extract key topics and generate comprehensive summaries from text discussions.</p>', unsafe_allow_html=True)

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
input_text = st.text_area(
    label="Enter your text for analysis:",
    height=250, 
    placeholder="Paste your text here (Reddit threads, forum discussions, articles, comments, etc.). For best results, include at least several paragraphs of content..."
)

# Configuration options
st.subheader("Analysis Options")
col1, col2 = st.columns(2)

with col1:
    is_reddit_content = st.checkbox(
        "Clean Reddit UI elements", 
        value=True,
        help="Enable this if your text contains Reddit posts/comments with UI elements like upvote counts, timestamps, usernames, etc. This will filter out non-content elements."
    )
    
    enable_semantic_refinement = st.checkbox(
        "Enable semantic refinement", 
        value=True,
        help="Uses advanced sentence embedding to improve content relevance. This produces higher quality results but may take slightly longer to process. Recommended for most text analysis."
    )
    
    # Add dev mode option for faster processing
    dev_mode = st.checkbox(
        "Dev mode (limit processing for speed)", 
        value=False,
        help="Limits the number of chunks processed for faster results. Use this for quick testing or when experiencing timeout issues."
    )

with col2:
    # Restore the num_topics slider to allow multiple topics
    num_topics = st.slider(
        "Number of topics to extract", 
        min_value=1, 
        max_value=5, 
        value=3,
        help="How many distinct topics to extract from the text. For shorter texts, use fewer topics. For longer, more diverse content, more topics may be appropriate."
    )
    
    max_sentences_per_topic = st.slider(
        "Max sentences per topic", 
        min_value=10, 
        max_value=50, 
        value=25,
        help="Maximum number of sentences to use for summarization per topic. Higher values provide more context but may take longer to process. Recommended: 20-30 for most discussions."
    )

# Advanced options in an expander
with st.expander("Advanced Summarization Options", expanded=False):
    # Display options
    st.markdown("#### Display Options")
    show_pre_summary_sentences = st.checkbox(
        "Show pre-summarization sentences", 
        value=False,
        help="Display the filtered sentences before they are summarized. Useful for understanding how sentences are selected for each topic and verifying content extraction."
    )
    
    show_chunk_summaries = st.checkbox(
        "Show per-chunk summaries", 
        value=False,
        help="Display summaries for each content chunk before final merging. This helps understand the two-pass summarization process and how different content sections are processed."
    )
    
    # Summarization configuration
    st.markdown("#### Summarization Settings")
    
    chunked_summarization = st.checkbox(
        "Use context-aware chunking", 
        value=True,
        help="Groups related sentences together for better coherence. This improves summary quality by maintaining context between related points and preventing fragmented summaries."
    )
    
    final_compression = st.checkbox(
        "Enable final summary compression (two-pass)", 
        value=True,
        help="Performs a second summarization pass to create a more concise and coherent final summary. This helps connect ideas across different chunks and removes redundancy."
    )
    
    if chunked_summarization:
        chunk_size = st.slider(
            "Chunk size", 
            min_value=5, 
            max_value=20, 
            value=10,
            help="Maximum sentences per chunk. Smaller chunks (5-8) focus on specific points while larger chunks (10-15) may capture broader context. Too large can reduce quality."
        )
    else:
        chunk_size = 100  # Effectively disable chunking
    
    # Custom prompt input
    st.markdown("#### Custom Prompt")
    use_custom_prompt = st.checkbox(
        "Use custom summarization prompt", 
        value=False,
        help="Provide a custom prompt for the summarization model to guide its output style and focus. Use this for specialized content or to control summary format and tone."
    )
    
    if use_custom_prompt:
        custom_prompt = st.text_area(
            "Custom Summarization Prompt:", 
            value="Summarize the following discussion about {topic} into 2-3 clear sentences:",
            help="You can use {topic} as a placeholder for the detected topic name. The model will replace this with the actual topic. Example: 'Provide key insights about {topic} in 3 bullet points.'"
        )
    else:
        custom_prompt = None
    
    # Topic name configuration
    st.markdown("#### Topic Configuration")
    use_topic_names = st.checkbox(
        "Set custom topic names", 
        value=False,
        help="Provide specific names for topics to improve summarization quality and relevance. Useful when you already know the expected topics in your content."
    )
    
    if use_topic_names:
        topic_name_template = st.text_input(
            "Topic name template:", 
            value="{topic}",
            help="Template for topic names. Use {topic} as placeholder for the generated topic name. Example: 'Medical: {topic}' will prefix all topics with 'Medical:'."
        )
    else:
        topic_name_template = "{topic}"  # Default format

analyze_button = st.button(
    "Analyze Text", 
    type="primary", 
    help="Click to process the text and generate summaries. Processing may take a few moments depending on text length and complexity."
)

# --- Analysis and Output ---
if analyze_button and input_text:
    if not input_text.strip():
        st.warning("Please enter some text to analyze.")
    else:
        # Create progress tracking elements
        progress_container = st.empty()
        with progress_container.container():
            st.write("Starting analysis...")
            progress_bar = st.progress(0)
            status_text = st.empty()
        
        with st.spinner("Analyzing text... This might take a moment depending on the text length and complexity."):
            try:
                # Prepare the request payload with enhanced options
                payload = json.dumps({
                    "text": input_text,
                    "is_reddit_content": is_reddit_content,
                    "num_topics": num_topics,
                    "enable_semantic_refinement": enable_semantic_refinement,
                    "max_sentences_per_topic": max_sentences_per_topic,
                    "chunked_summarization": chunked_summarization,
                    "final_compression": final_compression,
                    "chunk_size": chunk_size,
                    "use_topic_names": use_topic_names,
                    "topic_name_template": topic_name_template,
                    "custom_prompt": custom_prompt if use_custom_prompt else None,
                    "show_pre_summary_sentences": show_pre_summary_sentences,
                    "show_chunk_summaries": show_chunk_summaries,
                    "dev_mode": dev_mode  # Pass dev_mode to backend
                })
                headers = {'Content-Type': 'application/json'}

                # Update progress
                status_text.text("Sending request to backend...")
                progress_bar.progress(10)

                # Send request to backend
                response = requests.post(BACKEND_URL, data=payload, headers=headers, timeout=180) # Increased timeout for processing
                
                # Update progress
                status_text.text("Processing response...")
                progress_bar.progress(90)

                response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

                # Clear progress elements
                progress_container.empty()

                # Process the response
                results_data = response.json()
                results = results_data.get("results", [])

                st.subheader("Analysis Results")
                if results:
                    # Get the error message if present (for short text cases)
                    if "error" in results_data:
                        st.warning(results_data["error"])
                    
                    # Show backend message about topic count if available    
                    if "topic_count_info" in results_data:
                        st.info(f"⚠️ {results_data['topic_count_info']}. \n\n" +
                               "This can happen when:\n" +
                               "- The text doesn't contain enough distinct themes to form more topics\n" + 
                               "- There aren't enough sentences to distribute among more topics (aim for 3+ sentences per topic)\n" +
                               "- The content is focused on a limited set of closely related subjects")
                    # Otherwise check if we got fewer topics than requested
                    elif 1 <= len(results) < 1:
                        st.info(f"⚠️ Found {len(results)} topics instead of the requested {num_topics}. \n\n" +
                               "This can happen when:\n" +
                               "- The text doesn't contain enough distinct themes to form more topics\n" + 
                               "- There aren't enough sentences to distribute among more topics (aim for 3+ sentences per topic)\n" +
                               "- The content is focused on a limited set of closely related subjects")
                    
                    st.success(f"Found {len(results)} topics.")
                    
                    # Prepare data for export
                    export_data = {
                        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "topics": []
                    }
                    
                    for i, result in enumerate(results):
                        topic_name = result.get('topic', 'Unnamed Topic')
                        keywords = result.get('keywords', [])
                        keyword_text = ", ".join(keywords) if keywords else "No keywords available"
                        summary = result.get('summary', 'No summary available.')
                        
                        # Add to export data
                        export_data["topics"].append({
                            "name": topic_name,
                            "keywords": keywords,
                            "summary": summary
                        })
                        
                        # Display the topic with its contents
                        with st.expander(f"**Topic {i+1}: {topic_name}**", expanded=True):
                            # Create a row of keyword tags
                            if keywords:
                                st.write("Keywords:")
                                cols = st.columns(min(5, len(keywords)))
                                for j, keyword in enumerate(keywords):
                                    with cols[j % len(cols)]:
                                        st.markdown(f"<div class='keyword-tag'>{keyword}</div>", unsafe_allow_html=True)
                            
                            # Display the summary
                            st.markdown("### Summary")
                            st.markdown(f"<div class='summary-box'>{summary}</div>", unsafe_allow_html=True)
                            
                            # Add export button for individual topic summary
                            summary_text = f"TopicMind Summary - {topic_name}\n\nKeywords: {', '.join(keywords)}\n\n{summary}"
                            st.download_button(
                                label="📄 Export this topic",
                                data=summary_text,
                                file_name=f"summary_{topic_name.replace(' ', '_').lower()}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                                mime="text/plain",
                                help="Download just this topic's summary as a text file"
                            )
                            
                            # Display chunk summaries if enabled and available
                            if show_chunk_summaries and "chunk_summaries" in result and result["chunk_summaries"]:
                                st.markdown("### Chunk Summaries")
                                chunk_summaries = result.get("chunk_summaries", [])
                                
                                # Use tabs instead of expanders
                                chunk_tabs = st.tabs([f"Chunk {j+1}" for j in range(len(chunk_summaries))])
                                for j, (tab, chunk_summary) in enumerate(zip(chunk_tabs, chunk_summaries)):
                                    with tab:
                                        st.write(chunk_summary)
                                        
                                # Show if compression was applied
                                if result.get("final_compressed", False):
                                    st.caption("✓ Final compression was applied to merge these chunk summaries.")
                            
                            # Display pre-summarization sentences if enabled
                            if show_pre_summary_sentences and "source_sentences" in result:
                                st.markdown("### Source Sentences")
                                source_sentences = result.get("source_sentences", [])
                                
                                # Group sentences by chunks if chunking was used
                                if "chunks" in result and result["chunks"]:
                                    chunks = result.get("chunks", [])
                                    
                                    # Use tabs instead of expanders
                                    sentence_tabs = st.tabs([f"Chunk {j+1} Sentences" for j in range(len(chunks))])
                                    for j, (tab, chunk) in enumerate(zip(sentence_tabs, chunks)):
                                        with tab:
                                            for k, sent_dict in enumerate(chunk):
                                                sentence = sent_dict.get("text", "")
                                                source = sent_dict.get("source", "unknown")
                                                # Optional: Display relevance score if available
                                                score = sent_dict.get("relevance_score", None)
                                                score_text = f" (score: {score:.2f})" if score is not None else ""
                                                
                                                st.markdown(f"**{k+1}.** {sentence} <span style='color:gray; font-size:0.8em;'>ID: {source}{score_text}</span>", unsafe_allow_html=True)
                                else:
                                    # No chunks, display flat list
                                    for j, sent_dict in enumerate(source_sentences):
                                        sentence = sent_dict.get("text", "")
                                        source = sent_dict.get("source", "unknown")
                                        # Optional: Display relevance score if available
                                        score = sent_dict.get("relevance_score", None)
                                        score_text = f" (score: {score:.2f})" if score is not None else ""
                                        
                                        st.markdown(f"**{j+1}.** {sentence} <span style='color:gray; font-size:0.8em;'>ID: {source}{score_text}</span>", unsafe_allow_html=True)
                            
                            # Display evaluation scores if available
                            if "evaluation" in result:
                                evaluation = result.get("evaluation", {})
                                if "scores" in evaluation:
                                    st.markdown("### Quality Assessment")
                                    scores = evaluation["scores"]
                                    
                                    # Display scores in a more visual way
                                    score_cols = st.columns(len(scores))
                                    for j, (dimension, score) in enumerate(scores.items()):
                                        if score is not None and dimension != "overall":
                                            with score_cols[j]:
                                                st.metric(
                                                    label=dimension.capitalize(),
                                                    value=f"{score}/5",
                                                    delta=None
                                                )
                                    
                                    # Show overall score if available
                                    if "overall" in scores and scores["overall"] is not None:
                                        st.metric(
                                            label="Overall Quality",
                                            value=f"{scores['overall']:.1f}/5",
                                            delta=None
                                        )
                        
                        # Add divider between topics
                        st.divider()
                    
                    # Export functionality
                    st.subheader("Export Results")
                    export_format = st.radio("Export Format:", ["JSON", "Text"], horizontal=True)
                    
                    if export_format == "JSON":
                        json_data = json.dumps(export_data, indent=2)
                        st.download_button(
                            label="Download JSON Summary",
                            data=json_data,
                            file_name=f"topicmind_summary_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json",
                            help="Download all topics and summaries in JSON format"
                        )
                    else:  # Text format
                        text_data = f"TopicMind Summary - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                        for i, topic in enumerate(export_data["topics"]):
                            text_data += f"TOPIC {i+1}: {topic['name']}\n"
                            text_data += f"Keywords: {', '.join(topic['keywords'])}\n"
                            text_data += f"Summary: {topic['summary']}\n\n"
                            text_data += "-" * 50 + "\n\n"
                        
                        st.download_button(
                            label="Download Text Summary",
                            data=text_data,
                            file_name=f"topicmind_summary_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain",
                            help="Download all topics and summaries in plain text format"
                        )
                else:
                    st.info("No topics were identified in the provided text.")
                    st.markdown("""
                    **Possible reasons:**
                    - The text might be too short - try adding more content
                    - The text might not contain enough variation to form distinct topics
                    - You may need to adjust the number of topics (try a smaller number for shorter texts)
                    
                    **Try:**
                    - Adding more text
                    - If using Reddit content, make sure to check the "Clean Reddit UI elements" box
                    - Reduce the number of requested topics
                    """)

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
st.markdown("""
<h3>How to Use TopicMind</h3>
<ol>
    <li><strong>Input Text:</strong> Paste any discussion text (Reddit threads, forums, articles, etc.)</li>
    <li><strong>Configure Options:</strong> Adjust settings based on your content</li>
    <li><strong>Analyze:</strong> Click the button to extract topics and generate summaries</li>
    <li><strong>Review Results:</strong> Explore the topics and their summaries</li>
    <li><strong>Export:</strong> Download your results for sharing or future reference</li>
</ol>
""", unsafe_allow_html=True)

# Add version and contributor info at the bottom
st.markdown("---")
st.caption("TopicMind v1.0.0 | Smart Text Summarization | Open Source Project | © 2025")

# Add GitHub link
st.markdown("[![GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub&style=flat-square)](https://github.com/yourusername/topicmind)")
