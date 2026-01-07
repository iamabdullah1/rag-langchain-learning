# 🚀 RAG (Retrieval-Augmented Generation) with LangChain

A beginner-friendly tutorial series for building a **RAG system** from scratch using LangChain, ChromaDB, and Groq API.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![LangChain](https://img.shields.io/badge/LangChain-0.3-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 📖 What is RAG?

**RAG (Retrieval-Augmented Generation)** is a technique that combines:
- **Retrieval**: Finding relevant information from your documents
- **Generation**: Using AI to generate answers based on that information

```
Your Question  →  Search Documents  →  Get Context  →  AI Generates Answer
                    (Retrieval)                         (Generation)
```

## 🎯 What You'll Learn

By the end of this tutorial series, you'll understand how to:

1. **Load PDFs** into your application
2. **Split text** into manageable chunks
3. **Create embeddings** (convert text to numbers)
4. **Store vectors** in a database (ChromaDB)
5. **Build a RAG pipeline** with AI (Groq API)

## 📁 Project Structure

```
RAG-Langchain/
├── README.md                    # You are here!
├── requirements.txt             # Python dependencies
├── .env.example                 # Environment variables template
├── data/                        # Your PDF documents
│   ├── CodingqualitativedataResearchgate.pdf
│   └── EJ1172284.pdf
└── notebooks/
    ├── 01_document_loading.ipynb    # Step 1: Load PDFs
    ├── 02_text_chunking.ipynb       # Step 2: Split into chunks
    ├── 03_embeddings.ipynb          # Step 3: Create embeddings
    ├── 04_vector_store.ipynb        # Step 4: Store in ChromaDB
    └── 05_rag_complete.ipynb        # Step 5: Full RAG pipeline
```

## 🛠️ Setup Instructions

### Prerequisites
- Python 3.10+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

### Installation

1. **Clone this repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/rag-langchain-tutorial.git
   cd rag-langchain-tutorial
   ```

2. **Create a virtual environment**
   ```bash
   # Using uv (recommended)
   uv venv
   source .venv/bin/activate
   
   # Or using pip
   python -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   # Using uv
   uv pip install -r requirements.txt
   
   # Or using pip
   pip install -r requirements.txt
   ```

4. **Set up API keys**
   ```bash
   cp .env.example .env
   # Edit .env and add your Groq API key
   ```

5. **Get your free Groq API key**
   - Go to [console.groq.com/keys](https://console.groq.com/keys)
   - Create a free account
   - Generate an API key
   - Add it to your `.env` file

## 📚 Learning Path

Follow these notebooks in order:

| # | Notebook | Description | Time |
|---|----------|-------------|------|
| 1 | [Document Loading](notebooks/01_document_loading.ipynb) | Load PDFs using PyPDFLoader | 10 min |
| 2 | [Text Chunking](notebooks/02_text_chunking.ipynb) | Split documents into smaller pieces | 15 min |
| 3 | [Embeddings](notebooks/03_embeddings.ipynb) | Convert text to vectors | 15 min |
| 4 | [Vector Store](notebooks/04_vector_store.ipynb) | Store & search with ChromaDB | 20 min |
| 5 | [Complete RAG](notebooks/05_rag_complete.ipynb) | Build the full pipeline | 30 min |

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    RAG PIPELINE                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌─────────────────────────────────────────┐              │
│   │         ONE-TIME SETUP (Notebooks 1-4)  │              │
│   │                                         │              │
│   │   PDFs  →  Chunks  →  Embeddings  →  ChromaDB         │
│   └─────────────────────────────────────────┘              │
│                                                             │
│   ┌─────────────────────────────────────────┐              │
│   │         QUERY TIME (Notebook 5)         │              │
│   │                                         │              │
│   │   Question → Search → Context → AI → Answer           │
│   └─────────────────────────────────────────┘              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 🔧 Tech Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Framework** | LangChain | Orchestration |
| **PDF Loading** | PyPDFLoader | Read PDF files |
| **Embeddings** | HuggingFace (all-MiniLM-L6-v2) | Text → Vectors |
| **Vector DB** | ChromaDB | Store & search vectors |
| **LLM** | Groq API (llama-3.1-8b) | Generate answers |

## 💡 Key Concepts Explained

### What are Embeddings?
Embeddings convert text into numbers (vectors) that capture meaning:
```
"qualitative research" → [0.023, -0.051, 0.089, ...]  (384 numbers)
```
Similar texts have similar vectors, allowing us to find related content!

### What is a Vector Database?
A database optimized for storing and searching vectors:
- **ChromaDB**: Easy to use, auto-saves, great for learning
- **FAISS**: Faster for huge datasets, more manual work

### Why Chunking?
AI models have limited context windows. We split documents into smaller pieces (~500 characters) so we can:
1. Store them efficiently
2. Find the most relevant pieces
3. Fit them in the AI's context

## 🚀 Quick Start

After setup, run the complete RAG system:

```python
from notebooks.rag_utils import ask_my_documents

# Ask a question about your PDFs
result = ask_my_documents("What is qualitative data coding?")
print(result['answer'])
print(result['sources'])
```

## 📝 Example Output

```
❓ Question: What is qualitative data coding?

💬 Answer: Qualitative data coding is the process of organizing and 
categorizing qualitative data (like interview transcripts) by assigning 
labels or "codes" to segments of text. This helps researchers identify 
patterns, themes, and meanings in their data...

📚 Sources:
   1. CodingqualitativedataResearchgate.pdf, Page 4
   2. EJ1172284.pdf, Page 2
```

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Open issues for questions or bugs
- Submit pull requests for improvements
- Share your own RAG experiments

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [LangChain](https://langchain.com/) for the amazing framework
- [ChromaDB](https://www.trychroma.com/) for the vector database
- [Groq](https://groq.com/) for fast, free AI inference
- [HuggingFace](https://huggingface.co/) for open-source embeddings

---

**Happy Learning! 🎓**

If you found this helpful, please ⭐ star this repository!
