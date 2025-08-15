import streamlit as st
import os
import tempfile
import shutil
from pathlib import Path
from dotenv import load_dotenv

# Import custom modules
from src.github_loader import GitHubLoader
from src.code_parser import CodeParser
from src.chunking import CodeChunker
from src.embeddings import EmbeddingGenerator
from src.vector_store import VectorStore
from src.retriever import CodeRetriever
from src.generator import ResponseGenerator
from config.settings import settings

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Code Documentation RAG",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    pre {
        background-color: #2b2b2b;
        padding: 1rem;
        border-radius: 5px;
        overflow-x: auto;
    }
    .source-box {
        background-color: #f0f2f6;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'embeddings_generator' not in st.session_state:
    st.session_state.embeddings_generator = None
if 'retriever' not in st.session_state:
    st.session_state.retriever = None
if 'generator' not in st.session_state:
    st.session_state.generator = None
if 'repo_loaded' not in st.session_state:
    st.session_state.repo_loaded = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

def initialize_components():
    """Initialize RAG components"""
    if not st.session_state.embeddings_generator:
        st.session_state.embeddings_generator = EmbeddingGenerator(
            model_type="openai",
            model_name=settings.EMBEDDING_MODEL
        )
    
    if not st.session_state.vector_store:
        st.session_state.vector_store = VectorStore(
            collection_name=settings.COLLECTION_NAME,
            persist_directory=settings.PERSIST_DIRECTORY
        )
    
    parser = CodeParser()
    
    if not st.session_state.retriever:
        st.session_state.retriever = CodeRetriever(
            vector_store=st.session_state.vector_store,
            embeddings_generator=st.session_state.embeddings_generator,
            parser=parser
        )
    
    if not st.session_state.generator:
        st.session_state.generator = ResponseGenerator(
            model_name=settings.LLM_MODEL
        )
    
    return parser

def process_repository(repo_url: str, progress_callback=None):
    """Process a GitHub repository"""
    temp_dir = None
    try:
        # Initialize components
        parser = initialize_components()
        
        # Initialize GitHub loader
        loader = GitHubLoader(github_token=settings.GITHUB_TOKEN)
        
        # Update progress
        if progress_callback:
            progress_callback(0.1, "Cloning repository...")
        
        # Clone repository
        temp_dir = loader.clone_repository(repo_url)
        
        # Get repository info
        repo_info = loader.get_repository_info(repo_url)
        
        # Update progress
        if progress_callback:
            progress_callback(0.2, "Loading files...")
        
        # Load files
        all_extensions = list(settings.CODE_EXTENSIONS.keys()) + settings.DOC_EXTENSIONS
        documents = loader.load_repository_files(temp_dir, all_extensions)
        
        if not documents:
            raise ValueError("No supported files found in the repository")
        
        # Update progress
        if progress_callback:
            progress_callback(0.4, "Parsing and chunking code...")
        
        # Initialize chunker
        chunker = CodeChunker(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP
        )
        
        # Process documents
        all_chunks = []
        for i, doc in enumerate(documents):
            progress = 0.4 + (0.3 * (i / len(documents)))
            if progress_callback:
                progress_callback(progress, f"Processing file {i+1}/{len(documents)}...")
            
            # Choose chunking strategy based on file type
            if doc['metadata']['file_type'] in settings.CODE_EXTENSIONS:
                chunks = chunker.chunk_code(doc['content'], doc['metadata'])
            else:
                chunks = chunker.chunk_documentation(doc['content'], doc['metadata'])
            
            all_chunks.extend(chunks)
        
        # Create semantic chunks for code files
        if progress_callback:
            progress_callback(0.7, "Creating semantic chunks...")
        
        semantic_chunks = chunker.create_semantic_chunks(documents, parser)
        all_chunks.extend(semantic_chunks)
        
        # Update progress
        if progress_callback:
            progress_callback(0.8, "Generating embeddings and storing...")
        
        # Add to vector store
        st.session_state.vector_store.add_documents(
            all_chunks,
            st.session_state.embeddings_generator
        )
        
        # Update progress
        if progress_callback:
            progress_callback(1.0, "Repository processed successfully!")
        
        return {
            'success': True,
            'repo_info': repo_info,
            'total_files': len(documents),
            'total_chunks': len(all_chunks)
        }
    
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }
    
    finally:
        # Cleanup temporary directory
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

def main():
    st.title("🔍 Code Documentation RAG System")
    st.markdown("*Intelligent code understanding and documentation retrieval*")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key input
        api_key = st.text_input(
            "OpenAI API Key",
            value=os.getenv("OPENAI_API_KEY", ""),
            type="password",
            help="Enter your OpenAI API key"
        )
        
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
            settings.OPENAI_API_KEY = api_key
        
        st.divider()
        
        # Repository input
        st.header("📦 Repository Setup")
        repo_url = st.text_input(
            "GitHub Repository URL",
            placeholder="https://github.com/owner/repo",
            help="Enter the GitHub repository URL to analyze"
        )
        
        github_token = st.text_input(
            "GitHub Token (Optional)",
            value=os.getenv("GITHUB_TOKEN", ""),
            type="password",
            help="For private repositories or to avoid rate limits"
        )
        
        if github_token:
            settings.GITHUB_TOKEN = github_token
        
        if st.button("🚀 Load Repository", type="primary", disabled=not repo_url):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def update_progress(progress, status):
                progress_bar.progress(progress)
                status_text.text(status)
            
            result = process_repository(repo_url, update_progress)
            
            if result['success']:
                st.success(f"✅ Repository loaded successfully!")
                st.info(f"📊 Processed {result['total_files']} files into {result['total_chunks']} chunks")
                
                if result['repo_info']:
                    with st.expander("Repository Information"):
                        for key, value in result['repo_info'].items():
                            if value:
                                st.write(f"**{key.title()}:** {value}")
                
                st.session_state.repo_loaded = True
            else:
                st.error(f"❌ Error: {result['error']}")
        
        st.divider()
        
        # Settings
        with st.expander("🔧 Advanced Settings"):
            response_type = st.selectbox(
                "Response Type",
                ["explanation", "code_example", "api_reference", "implementation_guide"],
                help="Choose the type of response generation"
            )
            
            num_results = st.slider(
                "Number of Results",
                min_value=1,
                max_value=10,
                value=5,
                help="Number of relevant chunks to retrieve"
            )
            
            st.session_state.response_type = response_type
            st.session_state.num_results = num_results
        
        # Stats
        if st.session_state.vector_store:
            stats = st.session_state.vector_store.get_collection_stats()
            if stats:
                st.divider()
                st.metric("Documents in Store", stats.get('total_documents', 0))
    
    # Main content area
    if not api_key:
        st.warning("⚠️ Please enter your OpenAI API key in the sidebar to continue")
        st.stop()
    
    if not st.session_state.repo_loaded:
        # Welcome message
        st.info("👋 Welcome! Please load a GitHub repository from the sidebar to get started.")
        
        # Feature highlights
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            ### 🎯 Features
            - Multi-language support
            - Intelligent code parsing
            - Semantic search
            - Context-aware responses
            """)
        
        with col2:
            st.markdown("""
            ### 🔧 Capabilities
            - Code explanation
            - API documentation
            - Implementation guides
            - Example generation
            """)
        
        with col3:
            st.markdown("""
            ### 📚 Supported Languages
            - Python
            - JavaScript/TypeScript
            - Java, C/C++
            - Go, Rust, Ruby, PHP
            """)
    else:
        # Chat interface
        st.header("💬 Ask Questions About the Code")
        
        # Query input
        query = st.text_area(
            "Enter your question:",
            placeholder="e.g., How does the authentication system work? Show me examples of API endpoints.",
            height=100
        )
        
        col1, col2 = st.columns([1, 5])
        with col1:
            search_button = st.button("🔍 Search", type="primary", disabled=not query)
        with col2:
            clear_button = st.button("🗑️ Clear History")
        
        if clear_button:
            st.session_state.chat_history = []
            st.rerun()
        
        if search_button and query:
            with st.spinner("Searching and generating response..."):
                # Retrieve relevant documents
                retrieved_docs = st.session_state.retriever.retrieve(
                    query,
                    k=st.session_state.num_results
                )
                
                if retrieved_docs:
                    # Generate response
                    response = st.session_state.generator.generate_response(
                        query,
                        retrieved_docs,
                        response_type=st.session_state.response_type
                    )
                    
                    # Add to chat history
                    st.session_state.chat_history.append({
                        'query': query,
                        'response': response
                    })
                else:
                    st.warning("No relevant documents found. Try rephrasing your query.")
        
        # Display chat history
        if st.session_state.chat_history:
            st.divider()
            st.header("📝 Conversation History")
            
            for i, item in enumerate(reversed(st.session_state.chat_history)):
                with st.container():
                    # User query
                    st.markdown(f"**🧑 You:** {item['query']}")
                    
                    # Bot response
                    st.markdown("**🤖 Assistant:**")
                    
                    # Display formatted response
                    response_html = st.session_state.generator.format_response_with_citations(item['response'])
                    st.markdown(response_html, unsafe_allow_html=True)
                    
                    # Show sources
                    if item['response'].get('sources'):
                        with st.expander(f"📎 Sources ({len(item['response']['sources'])} files)"):
                            for source in item['response']['sources']:
                                st.markdown(f"""
                                <div class="source-box">
                                    <b>File:</b> {source['file']}<br>
                                    <b>Language:</b> {source['language']}<br>
                                    <b>Type:</b> {source['type']}
                                </div>
                                """, unsafe_allow_html=True)
                    
                    st.divider()

if __name__ == "__main__":
    main()