import unittest
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock

# Import components to test
from src.code_parser import CodeParser
from src.chunking import CodeChunker
from src.embeddings import EmbeddingGenerator
from src.vector_store import VectorStore
from src.retriever import CodeRetriever
from src.generator import ResponseGenerator

class TestCodeParser(unittest.TestCase):
    def setUp(self):
        self.parser = CodeParser()
    
    def test_parse_python_code(self):
        code = """
def hello_world():
    '''This is a docstring'''
    print("Hello, World!")

class MyClass:
    def __init__(self):
        self.value = 42
"""
        result = self.parser.parse_code(code, 'python')
        
        self.assertEqual(len(result['functions']), 1)
        self.assertEqual(result['functions'][0]['name'], 'hello_world')
        self.assertEqual(len(result['classes']), 1)
        self.assertEqual(result['classes'][0]['name'], 'MyClass')
    
    def test_parse_javascript_code(self):
        code = """
function greet(name) {
    console.log(`Hello, ${name}!`);
}

class Person {
    constructor(name) {
        this.name = name;
    }
}
"""
        result = self.parser._parse_with_regex(code, 'javascript')
        
        self.assertTrue(len(result['functions']) > 0)
        self.assertTrue(len(result['classes']) > 0)
    
    def test_extract_api_references(self):
        code = """
response = requests.get(url)
data = json.loads(response.text)
result = process_data(data)
"""
        api_refs = self.parser.extract_api_references(code, 'python')
        
        self.assertTrue(len(api_refs) > 0)
        self.assertTrue(any('requests.get' in ref['text'] for ref in api_refs))

class TestCodeChunker(unittest.TestCase):
    def setUp(self):
        self.chunker = CodeChunker(chunk_size=100, chunk_overlap=20)
    
    def test_chunk_code(self):
        code = "def test():\n    pass\n" * 50  # Create long code
        metadata = {'language': 'python', 'file_path': 'test.py'}
        
        chunks = self.chunker.chunk_code(code, metadata)
        
        self.assertTrue(len(chunks) > 1)
        self.assertEqual(chunks[0].metadata['chunk_index'], 0)
        self.assertEqual(chunks[0].metadata['language'], 'python')
    
    def test_chunk_documentation(self):
        doc = "# Header\n\nContent\n\n" * 30
        metadata = {'file_path': 'README.md', 'file_type': '.md'}
        
        chunks = self.chunker.chunk_documentation(doc, metadata)
        
        self.assertTrue(len(chunks) > 0)
        self.assertEqual(chunks[0].metadata['chunk_type'], 'documentation')
    
    def test_syntax_context_extraction(self):
        code = """
import numpy as np

def calculate_mean(data):
    return np.mean(data)

class DataProcessor:
    pass
"""
        context = self.chunker._extract_syntax_context(code, 'python')
        
        self.assertIn('import numpy as np', context['imports'])
        self.assertIn('calculate_mean', context['function_definitions'])
        self.assertIn('DataProcessor', context['class_definitions'])

class TestVectorStore(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.vector_store = VectorStore(
            collection_name="test_collection",
            persist_directory=self.temp_dir
        )
    
    def tearDown(self):
        # Clean up
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_collection_creation(self):
        stats = self.vector_store.get_collection_stats()
        self.assertEqual(stats['collection_name'], 'test_collection')
        self.assertEqual(stats['total_documents'], 0)
    
    @patch('src.vector_store.FAISS')  # Changed from Chroma to FAISS
    def test_add_documents(self, mock_faiss):
        from langchain.schema import Document
        
        docs = [
            Document(page_content="Test content", metadata={"test": "metadata"})
        ]
        
        # Create a mock embeddings generator
        mock_embeddings = Mock()
        mock_embeddings.model = Mock()
        
        # Mock FAISS.from_texts to return a mock store
        mock_store = Mock()
        mock_faiss.from_texts.return_value = mock_store
        
        # Add documents
        self.vector_store.add_documents(docs, mock_embeddings)
        
        # Verify that the store was created
        self.assertIsNotNone(self.vector_store.langchain_store)

class TestRetriever(unittest.TestCase):
    def setUp(self):
        self.mock_vector_store = Mock()
        self.mock_embeddings = Mock()
        self.mock_parser = Mock()
        
        self.retriever = CodeRetriever(
            self.mock_vector_store,
            self.mock_embeddings,
            self.mock_parser
        )
    
    def test_enhance_query(self):
        query = "python function"
        enhanced = self.retriever._enhance_query(query)
        
        self.assertIn("python", enhanced.lower())
        self.assertIn("function", enhanced.lower())
        self.assertTrue(len(enhanced) > len(query))
    
    def test_determine_strategy(self):
        self.assertEqual(self.retriever._determine_strategy("implement function"), "code_search")
        self.assertEqual(self.retriever._determine_strategy("API endpoint"), "api_search")
        self.assertEqual(self.retriever._determine_strategy("how to use"), "hybrid")
        self.assertEqual(self.retriever._determine_strategy("random query"), "general")

class TestResponseGenerator(unittest.TestCase):
    @patch('src.generator.ChatOpenAI')
    def setUp(self, mock_llm):
        self.mock_llm = mock_llm
        self.generator = ResponseGenerator()
    
    def test_prepare_context(self):
        from langchain.schema import Document
        
        docs = [
            Document(
                page_content="def test(): pass",
                metadata={'file_path': 'test.py', 'language': 'python'}
            )
        ]
        
        context = self.generator._prepare_context(docs)
        
        self.assertIn("test.py", context)
        self.assertIn("def test(): pass", context)
    
    def test_extract_sources(self):
        from langchain.schema import Document
        
        docs = [
            Document(page_content="", metadata={'file_path': 'file1.py', 'language': 'python'}),
            Document(page_content="", metadata={'file_path': 'file2.js', 'language': 'javascript'}),
            Document(page_content="", metadata={'file_path': 'file1.py', 'language': 'python'})
        ]
        
        sources = self.generator._extract_sources(docs)
        
        self.assertEqual(len(sources), 2)  # Deduplicated
        self.assertTrue(any(s['file'] == 'file1.py' for s in sources))
        self.assertTrue(any(s['file'] == 'file2.js' for s in sources))

if __name__ == '__main__':
    unittest.main()
