# TopicMind: Key Topic Identification and Summarization

TopicMind is an NLP system designed to identify the main topics within extensive text collections (like Reddit threads) and generate concise, topic-focused summaries. It helps users quickly grasp the core ideas buried in lengthy discussions by filtering out noise and highlighting essential themes.

## Problem Solved

Addresses the challenge of information overload in large text datasets (e.g., online forums, articles) by automatically extracting key topics and providing summaries for each, saving users time and effort.

## Features

*   **Topic Detection:** Uncovers latent themes in text using:
    *   Latent Dirichlet Allocation (LDA) for messy, user-generated content (refined with LLM assistance).
    *   BERT classification for cleaner, pre-labeled data.
*   **Topic-Based Summarization:** Generates abstractive summaries focused on specific identified topics using a BART-based model.
*   **Web Interface:** Provides a simple interface (built with Streamlit) to input text and view the analysis results.

## Tech Stack

*   **Backend:** Python, Flask
*   **NLP/ML:** scikit-learn (for LDA), transformers (for BART, potentially BERT), NLTK/spaCy (for preprocessing), OpenAI API (for topic refinement)
*   **Frontend:** Streamlit
*   **Data:** Primarily Reddit comment datasets.

_(Note: Specific libraries based on common implementations; please update `requirements.txt` as needed)_.

## Project Structure

```
topicMind/
├── .git/               # Git repository data
├── .gitignore          # Files ignored by Git
├── README.md           # This file
├── topicmind/          # Main application package
│   ├── __init__.py
│   ├── app.py          # Flask application (if used)
│   ├── data/           # Data files (e.g., sample JSON)
│   ├── frontend/       # Streamlit UI code (streamlit_app.py)
│   ├── models/         # ML models (LDA, BART)
│   ├── prompts/        # Prompts for LLM refinement
│   ├── utils/          # Utility scripts (preprocessing, refinement logic)
│   └── requirements.txt # Project dependencies
```

## Current Status (as of [Current Date - Please Update])

*   **Core Pipeline:** Initial versions of LDA topic modeling, BART summarization, text preprocessing, and LLM-based topic refinement are implemented.
*   **Data Handling:** Scripts for loading and preprocessing Reddit data are in place.
*   **Frontend:** A basic Streamlit interface (`topicmind/frontend/streamlit_app.py`) allows users to input text and trigger the analysis.
*   **Integration:** The components (preprocessing, topic modeling, summarization, refinement) are integrated into a preliminary pipeline.
*   **Next Steps:** Focus on evaluation (coherence, ROUGE scores), fine-tuning models, improving robustness, and potentially adding BERT classification for cleaner data sources.

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
    pip install -r topicmind/requirements.txt
    ```
    _(You might need to install additional libraries like `transformers`, `torch`, `nltk`, `openai`, `scikit-learn` if not already included in requirements.txt)_.

4.  **Run the Streamlit application:**
    ```bash
    streamlit run topicmind/frontend/streamlit_app.py
    ```

## Usage

1.  Navigate to the Streamlit URL provided after running the command above.
2.  Paste the text you want to analyze into the input text area.
3.  Click the "Analyze" button.
4.  View the identified topics and their corresponding summaries.

## Team

*   Sai Krishna Vishnumolakala (Pipeline Integration)
*   Harsha Reddy Palapala (Data Collection & Cleaning)
*   Gagana Vivekananda (LDA Implementation)
*   Bhavitha Kakumanu (BART Summarizer Fine-tuning)
*   Balakrishna Mangala (Evaluation & Testing)

## TODOs / Future Work

*   Implement evaluation metrics (Coherence, ROUGE).
*   Conduct thorough testing and performance analysis.
*   Fine-tune BART summarizer specifically for topic focus.
*   Implement BERT classifier for handling cleaner datasets.
*   Refine LLM prompts for topic refinement.
*   Improve error handling and logging.
*   Expand dataset usage. 