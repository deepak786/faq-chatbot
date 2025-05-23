# FAQ Chatbot

A streamlit-based chatbot that answers questions based on a FAQ database using LangChain and Ollama for local LLM inference.

## Features

- 🤖 Local LLM inference using Ollama
- 💾 Persistent vector storage using Chroma
- 🔍 Semantic search for finding relevant answers
- 🌐 Interactive web interface using Streamlit
- 💬 Chat history with expandable Q&A pairs

## Prerequisites

- Python 3.8 or higher
- [Ollama](https://ollama.com/) installed on your machine with the following models:
  - `llama3.2` - for generating responses
  - `nomic-embed-text` - for text embeddings

### Installing Ollama Models

After installing Ollama, run these commands to pull the required models:
```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

## Setup

1. Clone the repository:
```bash
git clone https://github.com/deepak786/faq-chatbot.git
cd faq_chatbot
```

2. Create a virtual environment and activate it:
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Running the Application

1. Make sure Ollama is running on your machine:
```bash
ollama serve
```

2. In a new terminal, start the Streamlit application:
```bash
streamlit run faq_chatbot.py
```