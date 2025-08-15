from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
from langchain.schema import Document
import re

class CodeChunker:
    def __init__(self, chunk_size: int = 1500, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitters = self._initialize_splitters()
    
    def _initialize_splitters(self) -> Dict[str, RecursiveCharacterTextSplitter]:
        """Initialize language-specific splitters"""
        splitters = {}
        
        # Python splitter
        splitters['python'] = RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        
        # JavaScript/TypeScript splitter
        splitters['javascript'] = RecursiveCharacterTextSplitter.from_language(
            language=Language.JS,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        splitters['typescript'] = splitters['javascript']
        
        # Other languages
        for lang in [Language.CPP, Language.GO, Language.JAVA, Language.PHP, Language.RUBY]:
            lang_name = str(lang).split('.')[-1].lower()
            splitters[lang_name] = RecursiveCharacterTextSplitter.from_language(
                language=lang,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
        
        # Default splitter for documentation
        splitters['default'] = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        
        return splitters
    
    def chunk_code(self, content: str, metadata: Dict[str, Any]) -> List[Document]:
        """Chunk code files intelligently based on language"""
        language = metadata.get('language', 'default')
        splitter = self.splitters.get(language, self.splitters['default'])
        
        # Create chunks
        chunks = splitter.split_text(content)
        
        # Create documents with enhanced metadata
        documents = []
        for i, chunk in enumerate(chunks):
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                'chunk_index': i,
                'total_chunks': len(chunks),
                'chunk_type': 'code',
                'has_functions': bool(re.search(r'def\s+\w+|function\s+\w+', chunk)),
                'has_classes': bool(re.search(r'class\s+\w+', chunk)),
                'lines_of_code': len(chunk.split('\n'))
            })
            
            # Add syntax context
            chunk_metadata['syntax_context'] = self._extract_syntax_context(chunk, language)
            
            documents.append(Document(
                page_content=chunk,
                metadata=chunk_metadata
            ))
        
        return documents
    
    def chunk_documentation(self, content: str, metadata: Dict[str, Any]) -> List[Document]:
        """Chunk documentation files (markdown, rst, txt)"""
        # Use markdown-aware splitting
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n## ", "\n### ", "\n#### ", "\n\n", "\n", " ", ""]
        )
        
        chunks = splitter.split_text(content)
        
        documents = []
        for i, chunk in enumerate(chunks):
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                'chunk_index': i,
                'total_chunks': len(chunks),
                'chunk_type': 'documentation',
                'has_code_blocks': bool(re.search(r'```[\s\S]*?```', chunk)),
                'has_headers': bool(re.search(r'^#+\s', chunk, re.MULTILINE))
            })
            
            # Extract section headers
            headers = re.findall(r'^(#+\s+.+)$', chunk, re.MULTILINE)
            if headers:
                chunk_metadata['section_headers'] = headers
            
            documents.append(Document(
                page_content=chunk,
                metadata=chunk_metadata
            ))
        
        return documents
    
    def _extract_syntax_context(self, chunk: str, language: str) -> Dict[str, Any]:
        """Extract syntax context from a code chunk"""
        context = {
            'imports': [],
            'function_definitions': [],
            'class_definitions': [],
            'variable_definitions': []
        }
        
        # Language-specific patterns
        if language == 'python':
            context['imports'] = re.findall(r'^(?:from|import)\s+[\w\.]+', chunk, re.MULTILINE)
            context['function_definitions'] = re.findall(r'^def\s+(\w+)', chunk, re.MULTILINE)
            context['class_definitions'] = re.findall(r'^class\s+(\w+)', chunk, re.MULTILINE)
        elif language in ['javascript', 'typescript']:
            context['imports'] = re.findall(r'^(?:import|const|let|var)\s+.*?(?:from|require)', chunk, re.MULTILINE)
            context['function_definitions'] = re.findall(r'(?:function|const|let|var)\s+(\w+)\s*=?\s*(?:\(|=>)', chunk)
            context['class_definitions'] = re.findall(r'class\s+(\w+)', chunk)
        elif language == 'java':
            context['imports'] = re.findall(r'^import\s+[\w\.]+', chunk, re.MULTILINE)
            context['function_definitions'] = re.findall(r'(?:public|private|protected)?\s*\w+\s+(\w+)\s*\(', chunk)
            context['class_definitions'] = re.findall(r'(?:public|private)?\s*class\s+(\w+)', chunk)
        
        return context
    
    def create_semantic_chunks(self, documents: List[Dict[str, Any]], parser) -> List[Document]:
        """Create semantic chunks based on code structure"""
        semantic_documents = []
        
        for doc in documents:
            if doc['metadata'].get('file_type') in ['.py', '.js', '.ts', '.java']:
                # Parse the code
                parsed = parser.parse_code(doc['content'], doc['metadata']['language'])
                
                # Create chunks for each function/class
                for func in parsed['functions']:
                    if isinstance(func, dict) and 'name' in func:
                        func_content = self._extract_function_content(doc['content'], func)
                        if func_content:
                            metadata = doc['metadata'].copy()
                            metadata.update({
                                'chunk_type': 'function',
                                'function_name': func['name'],
                                'docstring': func.get('docstring', ''),
                                'semantic_type': 'function_definition'
                            })
                            semantic_documents.append(Document(
                                page_content=func_content,
                                metadata=metadata
                            ))
                
                for cls in parsed['classes']:
                    if isinstance(cls, dict) and 'name' in cls:
                        class_content = self._extract_class_content(doc['content'], cls)
                        if class_content:
                            metadata = doc['metadata'].copy()
                            metadata.update({
                                'chunk_type': 'class',
                                'class_name': cls['name'],
                                'docstring': cls.get('docstring', ''),
                                'methods': cls.get('methods', []),
                                'semantic_type': 'class_definition'
                            })
                            semantic_documents.append(Document(
                                page_content=class_content,
                                metadata=metadata
                            ))
        
        return semantic_documents
    
    def _extract_function_content(self, code: str, func_info: Dict) -> str:
        """Extract function content from code"""
        if 'line_start' in func_info and 'line_end' in func_info:
            lines = code.split('\n')
            start = max(0, func_info['line_start'] - 1)
            end = min(len(lines), func_info['line_end'])
            return '\n'.join(lines[start:end])
        return ""
    
    def _extract_class_content(self, code: str, class_info: Dict) -> str:
        """Extract class content from code"""
        if 'line_start' in class_info and 'line_end' in class_info:
            lines = code.split('\n')
            start = max(0, class_info['line_start'] - 1)
            end = min(len(lines), class_info['line_end'])
            return '\n'.join(lines[start:end])
        return ""