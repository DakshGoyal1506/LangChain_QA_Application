<!-- filepath: /C:/Users/daksh/Desktop/ZTM/Projects/LangChain_QA_Application/README.md -->

# LangChain_QA_Application

A question-answering pipeline built with LangChain, from initial experiments in Jupyter notebooks to a Streamlit UI and a final deployable app.

## Table of Contents

- [Overview](#overview)  
- [Project Structure](#project-structure)  
- [Getting Started](#getting-started)  
- [Workflow](#workflow)  
  - [01.ipynb](#01ipynb)  
  - [02.ipynb](#02ipynb)  
  - [project.ipynb](#projectipynb)  
  - [Streamlit App](#streamlit-app)  
  - [Final App](#final-app)  
- [Dependencies](#dependencies)  
- [License](#license)  

## Overview

This repository demonstrates a LangChain-powered QA system:
1. Prototype simple prompt→LLM chains  
2. Build embedding-based similarity search  
3. Assemble a full RAG (Retrieval-Augmented Generation) pipeline  
4. Expose it via Streamlit for interactive use  
5. Package into a standalone final application  

## Project Structure

```
LangChain_QA_Application/
├── 01.ipynb             # Initial LLM prompt & chain experiments
├── 02.ipynb             # Embeddings / similarity search demos
├── project.ipynb        # Full RAG pipeline prototype
├── streamlit_app.py     # Streamlit interface for RAG QA
├── app.py               # Final application entry point
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

## Getting Started

1. Clone this repo  
   ```bash
   git clone https://github.com/<USERNAME>/LangChain_QA_Application.git
   cd LangChain_QA_Application
   ```
2. Create & activate a virtual environment  
   ```bash
   python -m venv .venv
   .venv\Scripts\activate       # on Windows
   ```
3. Install dependencies  
   ```bash
   pip install -r requirements.txt
   ```

## Workflow

### 01.ipynb
- Explore basic LangChain `PromptTemplate`, `LLMChain`, and sequential chains.  
- Test simple question → answer flows using ChatOpenAI.

### 02.ipynb
- Implement embeddings-based retrieval (e.g., with Chroma).  
- Build a similarity search over document chunks.  

### project.ipynb
- Combine retrieval + LLM in a Retrieval-Augmented Generation (RAG) pipeline.  
- Format prompts, manage context windows, and handle multi-chain nesting.

### Streamlit App
- `streamlit_app.py`  
- Provide a web UI for user questions.  
- Perform document loading, chunking, embedding lookup, and LLM invocation on the fly.  
- Run with:
  ```bash
  streamlit run streamlit_app.py
  ```

### Final App
- `app.py`  
- Wrap up the entire pipeline into a console or API-driven application.  
- Entry point for deployment or containerization.

## Dependencies

- Python 3.8+  
- langchain  
- openai  
- chromadb (or your chosen vector store)  
- streamlit  
- numpy, pandas, tiktoken, etc.  

_(See `requirements.txt` for full list.)_

## License

MIT © 2025 Your Name  