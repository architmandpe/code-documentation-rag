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
```bash
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

Key Technologies

LangChain: Orchestration framework
ChromaDB: Vector database for embeddings
OpenAI: Embeddings and LLM generation
Tree-sitter: Multi-language code parsing
Streamlit: Web interface
PyGithub: GitHub API integration

🔧 Configuration
Edit config/settings.py to customize:

Embedding models
LLM models
Chunk sizes
Retrieval parameters
Supported languages and file types

📊 Evaluation Metrics
The system includes evaluation capabilities:

Retrieval Accuracy: Precision and recall of retrieved chunks
Response Quality: RAGAS metrics for generation quality
Latency: Query response time measurement
Coverage: Percentage of repository successfully indexed

🚢 Deployment
Option 1: Streamlit Cloud

Push to GitHub
Connect to Streamlit Cloud
Add secrets in dashboard
Deploy

Option 2: Docker
bashdocker build -t code-doc-rag .
docker run -p 8501:8501 --env-file .env code-doc-rag

Option 3: HuggingFace Spaces

Create new Space
Upload files
Configure secrets
Enable Streamlit SDK

code-doc-rag/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── config/
│   └── settings.py       # Configuration settings
├── src/
│   ├── github_loader.py  # Repository loading
│   ├── code_parser.py    # Code parsing logic
│   ├── chunking.py       # Document chunking
│   ├── embeddings.py     # Embedding generation
│   ├── vector_store.py   # Vector DB operations
│   ├── retriever.py      # RAG retrieval
│   └── generator.py      # Response generation
└── tests/
    └── test_components.py # Unit tests

🧪 Testing
Run tests:
bashpytest tests/
🤝 Contributing

Fork the repository
Create feature branch (git checkout -b feature/amazing-feature)
Commit changes (git commit -m 'Add amazing feature')
Push to branch (git push origin feature/amazing-feature)
Open Pull Request

📄 License
MIT License - see LICENSE file for details
🙏 Acknowledgments

OpenAI for GPT and embeddings
LangChain community
Tree-sitter maintainers
Streamlit team

📧 Contact
For questions or support, please open an issue on GitHub.

### 12. Environment Variables Template (`.env.example`)

```env
# OpenAI Configuration
OPENAI_API_KEY=your-openai-api-key-here

# GitHub Configuration (Optional)
GITHUB_TOKEN=your-github-token-here

# Model Configuration
EMBEDDING_MODEL=text-embedding-3-small
LLM_MODEL=gpt-4-turbo-preview

# Chunking Configuration
CHUNK_SIZE=1500
CHUNK_OVERLAP=200

# Vector Store Configuration
COLLECTION_NAME=code_documentation
PERSIST_DIRECTORY=./chroma_db

# Retrieval Configuration
TOP_K=5
SIMILARITY_THRESHOLD=0.7