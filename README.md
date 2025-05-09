# TopicMind: Key Topic Identification and Summarization

TopicMind is an NLP system designed to identify the main topics within extensive text collections (like Reddit threads) and generate concise, topic-focused summaries. It helps users quickly grasp the core ideas buried in lengthy discussions by filtering out noise and highlighting essential themes.

## Problem Solved

Addresses the challenge of information overload in large text datasets (e.g., online forums, articles) by automatically extracting key topics and providing summaries for each, saving users time and effort.

## Features

*   **Topic Detection:** Uncovers latent themes in text using:
    *   Fine-tuned BERTopic model (based on "all-mpnet-base-v2") trained on 300k+ Reddit comments from our `kaggle_RC_2019-05.csv` dataset.
    *   Sophisticated clustering and embedding techniques for semantically coherent topic grouping.
*   **Topic Name Refinement:** Uses OpenAI API to generate human-friendly topic names from the keywords extracted by our fine-tuned model.
*   **Topic-Based Summarization:** Generates abstractive summaries focused on specific identified topics using BART (facebook/bart-large-cnn). Fine-tuning on Reddit content is in progress, with plans to use the more efficient `sshleifer/distilbart-cnn-12-6` model.
*   **Web Interface:** Provides a simple interface (built with Streamlit) to input text and view the analysis results.
*   **Mac M1 Compatibility:** Specifically optimized for both GPU and non-GPU environments (including Apple Silicon).

## Evolution from Previous Approach

Our system has evolved significantly from its initial implementation:

### Previous Approach
- Used **Latent Dirichlet Allocation (LDA)** for topic modeling, which struggled with coherence in conversational text
- Relied heavily on generic models without domain-specific fine-tuning
- Faced compatibility issues on Apple Silicon hardware
- Used a single-stage summarization process that didn't properly separate topics

### Current Approach
- Implemented **BERTopic with advanced embeddings (all-mpnet-base-v2)** for dramatically improved topic coherence
- **Fine-tuned BERTopic model on 300k+ Reddit comments** for domain adaptation to social media content
- Created a **Mac-optimized architecture** with specific code paths for CPU-only and future MPS (Metal Performance Shaders) support
- Developed a **two-stage process**:
  1. Topic identification and keyword extraction (BERTopic)
  2. Per-topic summarization (BART)
- Added specialized **Reddit content preprocessing** to clean UI elements and formatting

### Why These Changes Matter
- **Topic Quality:** The move from LDA to fine-tuned BERTopic provides significantly more coherent and interpretable topics
- **Hardware Compatibility:** Optimization for Apple Silicon ensures the system works on modern Macs without GPU requirements
- **Focused Summarization:** By clustering similar sentences first, summaries are more focused on specific topics rather than mixing themes
- **Efficient API Usage:** OpenAI is used only for the specialized task of topic naming, reducing API costs and dependencies

## Tech Stack

*   **Backend:** Python, Flask
*   **NLP/ML:** 
    *   Fine-tuned BERTopic (with sentence-transformers) for topic modeling
    *   BART (facebook/bart-large-cnn) for topic-specific summarization (fine-tuning planned with `sshleifer/distilbart-cnn-12-6`)
    *   NLTK (for preprocessing)
    *   OpenAI API (only for topic name refinement, not for core analysis)
*   **Frontend:** Streamlit
*   **Data:** Trained on over 300k Reddit comments from the Kaggle Reddit Comments dataset (May 2019).

## Project Structure

```
topicMind/
├── app.py                 # Flask backend application
├── requirements.txt       # Project dependencies
├── README.md              # This file
├── .env                   # Environment variables (contains OpenAI API key)
├── BERTopic Model/        # Pre-trained models
│   ├── bertopic_model_300k_all-mpnet-base-v2  # Fine-tuned BERTopic model
│   └── bertopic_model_paraphrase-MiniLM-L12-v2  # Alternative model (smaller)
├── data/                  # Training and sample data
│   ├── kaggle_RC_2019-05.csv  # Reddit Comments dataset used for training
│   └── reddit_sample.json     # Sample data for testing
├── frontend/              # Streamlit UI code
│   └── streamlit_app.py   # Streamlit frontend application
├── models/                # Model loading and inference
│   ├── bertopic_model_simple.py  # BERTopic model loading & inference
│   └── bart_summarizer.py        # BART summarization logic
├── prompts/               # Prompts for LLM refinement
│   └── refine_topic.gpt.txt  # Prompt template for OpenAI topic refinement
└── utils/                 # Utility scripts
    ├── preprocessor.py    # Text preprocessing logic
    └── topic_refiner.py   # OpenAI integration for topic refinement
```

## OpenAI Integration - Topic Refinement Only

It's important to note that OpenAI API is used **only for topic name refinement**, not for the core analysis:

1. Our fine-tuned BERTopic model extracts keywords and groups sentences
2. BART model generates the summaries (transitioning to `sshleifer/distilbart-cnn-12-6` for efficiency)
3. OpenAI API receives only the keywords from each topic to generate a human-friendly topic name

The prompt template used for OpenAI can be found in `prompts/refine_topic.gpt.txt`:

```
You are an expert at analyzing and naming topics.

Given the following list of keywords that represent a single topic: {keywords}

Please provide a SHORT, clear, and concise name for this topic (3-5 words maximum). 
The name should accurately capture the essence of what these keywords collectively represent.

Your response should be ONLY the topic name, with no additional explanation or commentary.
```

This approach ensures:
- We get human-friendly topic names 
- We maintain control over the core analysis using our models
- The OpenAI API is used efficiently (small prompts, minimal tokens)

## Frontend Interface (Streamlit)

This section outlines the inputs and outputs for the Streamlit frontend.

**Input:**

1.  **Primary Input:**
    *   **Text Data:** A main text area for pasting large blocks of text (e.g., forum threads, articles).
2.  **Configuration Parameters:**
    *   **Number of Topics:** Slider to select desired number of topics (1-10).
    *   **Clean Reddit UI:** Option to clean Reddit-specific UI elements from input.

**Output:**

1.  **Identified Topics:**
    *   Display a list of identified topic labels, refined using OpenAI.
2.  **Topic Summaries:**
    *   Show the concise, abstractive summary generated by BART model for each corresponding topic.
3.  **Keywords:**
    *   Display the top keywords that represent each topic.

## Dataset Information

The models were fine-tuned using the Kaggle Reddit Comments dataset from May 2019 (`data/kaggle_RC_2019-05.csv`), containing over 300,000 comments from various subreddits. This dataset was chosen because:

1. It contains diverse topics and writing styles
2. It represents real-world conversational content
3. It includes challenging linguistic patterns (slang, abbreviations, nested discussions)

The dataset was preprocessed to remove noise, clean Reddit-specific formatting, and normalize text before being used for fine-tuning.

## Getting Started

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/SaiKrishna-KK/topic-mind.git
    cd topicMind
    ```
2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    NOTE: For Mac M1/M2 users, ensure PyTorch is installed for CPU:
    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    ```

4.  **Set up environment variables:**
    Create a `.env` file in the project root with your OpenAI API key:
    ```
    OPENAI_API_KEY=your_openai_api_key_here
    ```

5.  **Run the application:**
    
    **Option 1:** Use the all-in-one launcher script:
    ```bash
    ./run_topicmind.sh
    ```
    This script:
    - Starts the backend and shows dependency loading (PyTorch, TensorFlow, etc.)
    - Waits for the backend to fully initialize and checks its health
    - Only starts the frontend after confirming the backend is ready
    - Provides clean shutdown with Ctrl+C
    
    **Option 2:** Start services separately:
    ```bash
    # Start the backend
    python app.py
    
    # In a separate terminal (after backend is fully initialized)
    streamlit run frontend/streamlit_app.py
    ```

## Usage

1.  Navigate to the Streamlit URL provided after running the command above (typically http://localhost:8501).
2.  Paste the text you want to analyze into the input text area.
3.  Adjust the "Number of topics" slider as needed.
4.  Check "Clean Reddit UI elements" if analyzing Reddit content.
5.  Click the "Analyze" button.
6.  View the identified topics and their corresponding summaries.

## Mac M1/M2 Compatibility

TopicMind has been specifically optimized to work on Apple Silicon (M1/M2) Macs:

* All PyTorch operations run in CPU-only mode
* The BERTopic model is loaded in a way that's compatible with CPU environments
* The application will automatically detect and handle the non-CUDA environment
* Future releases will include MPS (Metal Performance Shaders) support for GPU acceleration

## Team

*   Sai Krishna Vishnumolakala (Pipeline Integration)
*   Harsha Reddy Palapala (Data Collection & Cleaning)
*   Gagana Vivekananda (BERTopic Implementation & Fine-tuning)
*   Bhavitha Kakumanu (BART Summarizer Fine-tuning)
*   Balakrishna Mangala (Evaluation & Testing)

## TODOs / Future Work

**Immediate Priorities:**

1.  **GPU Optimization for Apple Silicon:**
    *   Implement MPS (Metal Performance Shaders) support for M1/M2 Macs
    *   Optimize model loading and inference for GPU acceleration
2.  **BART Fine-tuning:**
    *   Use `sshleifer/distilbart-cnn-12-6` instead of facebook/bart-large-cnn (80% as strong, but 50% faster to train)
    *   Fine-tune the lightweight DistilBART model on Reddit content for more accurate, contextually appropriate summaries
    *   Leverage the model's zero-shot capabilities for minimal latency in production

**Next Steps:**

*   **UI Improvements:** 
    *   Add topic visualization (word clouds, topic weights)
    *   Implement file upload functionality
*   **Model Enhancements:**
    *   Further improve performance through hyperparameter optimization
    *   Explore more advanced clustering techniques beyond K-means

**Future Enhancements:**

*   Add multi-language support for global content analysis
*   Develop a local alternative to OpenAI for topic refinement
*   Chrome extension based page ingestion and auto-scrapping rather than copy-paste. 
*   Create a more interactive interface for exploring topic relationships 