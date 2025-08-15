from typing import List, Dict, Any, Optional
from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import re
from pygments import highlight
from pygments.lexers import get_lexer_by_name
from pygments.formatters import HtmlFormatter

class ResponseGenerator:
    def __init__(self, model_name: str = "gpt-4-turbo-preview"):
        from config.settings import settings
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0.2,
            openai_api_key=settings.OPENAI_API_KEY
        )
        self.formatter = HtmlFormatter(style='monokai')
    
    def generate_response(self, query: str, retrieved_docs: List[Document], 
                         response_type: str = "explanation") -> Dict[str, Any]:
        """Generate response based on retrieved documents"""
        
        if response_type == "code_example":
            return self._generate_code_example(query, retrieved_docs)
        elif response_type == "api_reference":
            return self._generate_api_reference(query, retrieved_docs)
        elif response_type == "implementation_guide":
            return self._generate_implementation_guide(query, retrieved_docs)
        else:
            return self._generate_explanation(query, retrieved_docs)
    
    def _generate_explanation(self, query: str, docs: List[Document]) -> Dict[str, Any]:
        """Generate a detailed explanation"""
        context = self._prepare_context(docs)
        
        prompt = ChatPromptTemplate.from_template("""
        You are a helpful coding assistant with expertise in software development and documentation.
        
        Based on the following code and documentation context, provide a clear and detailed explanation 
        for the user's query. Include relevant code examples from the context when appropriate.
        
        Context:
        {context}
        
        User Query: {query}
        
        Please provide:
        1. A clear explanation addressing the query
        2. Relevant code snippets from the context (if applicable)
        3. Best practices or important considerations
        4. Links to related functions or classes mentioned in the context
        
        Format your response in markdown with proper code blocks.
        """)
        
        response = self.llm.invoke(
            prompt.format_messages(context=context, query=query)
        )
        
        return {
            'response': response.content,
            'type': 'explanation',
            'sources': self._extract_sources(docs)
        }
    
    def _generate_code_example(self, query: str, docs: List[Document]) -> Dict[str, Any]:
        """Generate code examples based on context"""
        context = self._prepare_context(docs)
        
        # Determine the primary language from docs
        languages = [doc.metadata.get('language', 'unknown') for doc in docs]
        primary_language = max(set(languages), key=languages.count) if languages else 'python'
        
        prompt = ChatPromptTemplate.from_template("""
        You are an expert programmer. Based on the following code context, 
        generate a practical code example that demonstrates the answer to the user's query.
        
        Context:
        {context}
        
        User Query: {query}
        
        Primary Language: {language}
        
        Please provide:
        1. A working code example that directly addresses the query
        2. Inline comments explaining key parts
        3. Brief explanation of how the code works
        4. Any necessary imports or setup
        
        Format the code with proper syntax highlighting for {language}.
        """)
        
        response = self.llm.invoke(
            prompt.format_messages(
                context=context, 
                query=query, 
                language=primary_language
            )
        )
        
        # Extract and highlight code blocks
        highlighted_response = self._highlight_code_blocks(response.content, primary_language)
        
        return {
            'response': highlighted_response,
            'type': 'code_example',
            'language': primary_language,
            'sources': self._extract_sources(docs)
        }
    
    def _generate_api_reference(self, query: str, docs: List[Document]) -> Dict[str, Any]:
        """Generate API reference documentation"""
        context = self._prepare_context(docs)
        
        prompt = ChatPromptTemplate.from_template("""
        You are a technical documentation expert. Based on the following code context,
        generate comprehensive API reference documentation for the requested functionality.
        
        Context:
        {context}
        
        User Query: {query}
        
        Please provide API documentation including:
        1. Function/Class signature
        2. Parameters (name, type, description)
        3. Return values (type, description)
        4. Usage examples
        5. Related functions or methods
        6. Exceptions or error cases
        
        Format the response as structured API documentation.
        """)
        
        response = self.llm.invoke(
            prompt.format_messages(context=context, query=query)
        )
        
        return {
            'response': response.content,
            'type': 'api_reference',
            'sources': self._extract_sources(docs)
        }
    
    def _generate_implementation_guide(self, query: str, docs: List[Document]) -> Dict[str, Any]:
        """Generate step-by-step implementation guide"""
        context = self._prepare_context(docs)
        
        prompt = ChatPromptTemplate.from_template("""
        You are a senior software engineer creating implementation guides.
        Based on the following code context, create a detailed implementation guide.
        
        Context:
        {context}
        
        User Query: {query}
        
        Please provide:
        1. Overview of the implementation approach
        2. Step-by-step implementation guide
        3. Code snippets for each step
        4. Common pitfalls and how to avoid them
        5. Testing considerations
        6. Performance optimization tips (if relevant)
        
        Format as a structured guide with clear sections and code examples.
        """)
        
        response = self.llm.invoke(
            prompt.format_messages(context=context, query=query)
        )
        
        return {
            'response': response.content,
            'type': 'implementation_guide',
            'sources': self._extract_sources(docs)
        }
    
    def _prepare_context(self, docs: List[Document]) -> str:
        """Prepare context from retrieved documents"""
        context_parts = []
        
        for i, doc in enumerate(docs):
            metadata = doc.metadata
            
            # Create context header
            header = f"[Source {i+1}: {metadata.get('file_path', 'Unknown')}]"
            if metadata.get('function_name'):
                header += f" - Function: {metadata['function_name']}"
            elif metadata.get('class_name'):
                header += f" - Class: {metadata['class_name']}"
            
            # Add the content
            context_parts.append(f"{header}\n```{metadata.get('language', '')}\n{doc.page_content}\n```\n")
        
        return "\n".join(context_parts)
    
    def _extract_sources(self, docs: List[Document]) -> List[Dict[str, Any]]:
        """Extract source information from documents"""
        sources = []
        seen = set()
        
        for doc in docs:
            file_path = doc.metadata.get('file_path', 'Unknown')
            if file_path not in seen:
                seen.add(file_path)
                sources.append({
                    'file': file_path,
                    'language': doc.metadata.get('language', 'unknown'),
                    'type': doc.metadata.get('chunk_type', 'unknown')
                })
        
        return sources
    
    def _highlight_code_blocks(self, text: str, language: str) -> str:
        """Apply syntax highlighting to code blocks"""
        # Find all code blocks
        code_block_pattern = r'```(\w*)\n(.*?)\n```'
        
        def highlight_match(match):
            lang = match.group(1) or language
            code = match.group(2)
            
            try:
                lexer = get_lexer_by_name(lang, stripall=True)
                highlighted = highlight(code, lexer, self.formatter)
                return f'<div class="highlight-wrapper">{highlighted}</div>'
            except:
                return f'<pre><code>{code}</code></pre>'
        
        highlighted_text = re.sub(code_block_pattern, highlight_match, text, flags=re.DOTALL)
        return highlighted_text
    
    def format_response_with_citations(self, response: Dict[str, Any]) -> str:
        """Format response with proper citations"""
        formatted = response['response']
        
        # Add sources section
        if response.get('sources'):
            formatted += "\n\n---\n### Sources\n"
            for i, source in enumerate(response['sources'], 1):
                formatted += f"{i}. `{source['file']}` ({source['language']})\n"
        
        return formatted