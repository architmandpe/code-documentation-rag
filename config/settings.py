# config/settings.py
import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
    
    # Model Settings
    EMBEDDING_MODEL = "text-embedding-3-small"
    LLM_MODEL = "gpt-4-turbo-preview"
    
    # Chunking Settings
    CHUNK_SIZE = 1500
    CHUNK_OVERLAP = 200
    
    # Vector Store Settings - Updated for FAISS
    COLLECTION_NAME = "code_documentation"
    PERSIST_DIRECTORY = "./faiss_index"  # Changed from ./chroma_db
    
    # Retrieval Settings
    TOP_K = 5
    SIMILARITY_THRESHOLD = 0.7
    
    # Supported Languages
    SUPPORTED_LANGUAGES = [
        'python', 'javascript', 'typescript', 'java', 
        'cpp', 'c', 'go', 'rust', 'ruby', 'php'
    ]
    
    # File Extensions
    CODE_EXTENSIONS = {
        '.py': 'python',
        '.js': 'javascript',
        '.ts': 'typescript',
        '.java': 'java',
        '.cpp': 'cpp',
        '.c': 'c',
        '.go': 'go',
        '.rs': 'rust',
        '.rb': 'ruby',
        '.php': 'php'
    }
    
    DOC_EXTENSIONS = ['.md', '.rst', '.txt', '.doc']

settings = Settings()
