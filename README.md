# 📚 Code Documentation RAG System

An intelligent Retrieval-Augmented Generation (RAG) system for software documentation that understands code snippets, API references, and provides contextual code examples with implementation details.

## 🎯 Features

- **Multi-Language Support**: Parse and understand code in Python, JavaScript, TypeScript, Java, C/C++, Go, Rust, Ruby, and PHP
- **Intelligent Code Parsing**: Extract functions, classes, imports, and documentation using AST and tree-sitter
- **Semantic Chunking**: Smart code-aware chunking that preserves context and structure
- **Hybrid Search**: Combines semantic and keyword search for optimal retrieval
- **Context-Aware Generation**: Generate explanations, code examples, API docs, and implementation guides
- **GitHub Integration**: Direct repository cloning and processing
- **Syntax Highlighting**: Beautiful code formatting with Pygments
- **Interactive UI**: Clean Streamlit interface with conversation history

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- OpenAI API key
- GitHub token (optional, for private repos)

### Installation

1. Clone the repository:
bash
git clone https://github.com/yourusername/code-doc-rag.git
cd code-doc-rag

2. Install dependencies:

bash
pip install -r requirements.txt

3. Set up environment variables:

bash
cp .env.example .env
# Edit .env and add your API keys

4. Run the application:

bashstreamlit run app.py

📖 Usage

Enter API Keys: Add your OpenAI API key in the sidebar
Load Repository: Enter a GitHub repository URL and click "Load Repository"
Ask Questions: Type your questions about the code in the main interface
Get Answers: Receive detailed explanations, code examples, or API documentation

Example Queries

"How does the authentication system work?"
"Show me examples of database connections"
"What are the API endpoints for user management?"
"Explain the main class architecture"
"How to implement error handling in this project?"

┌─────────────────┐     ┌──────────────┐     ┌──────────────┐
│  GitHub Loader  │────▶│ Code Parser  │────▶│   Chunker    │
└─────────────────┘     └──────────────┘     └──────────────┘
                                                      │
                                                      ▼
┌─────────────────┐     ┌──────────────┐     ┌──────────────┐
│    Generator    │◀────│  Retriever   │◀────│ Vector Store │
└─────────────────┘     └──────────────┘     └──────────────┘
