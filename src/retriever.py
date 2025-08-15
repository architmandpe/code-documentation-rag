from typing import List, Dict, Any, Optional
from langchain.schema import Document
import re

class CodeRetriever:
    def __init__(self, vector_store, embeddings_generator, parser):
        self.vector_store = vector_store
        self.embeddings_generator = embeddings_generator
        self.parser = parser
    
    def retrieve(self, query: str, k: int = 5, filters: Optional[Dict] = None) -> List[Document]:
        """Main retrieval method with query enhancement"""
        # Enhance query for code search
        enhanced_query = self._enhance_query(query)
        
        # Determine retrieval strategy
        strategy = self._determine_strategy(query)
        
        if strategy == "code_search":
            return self._retrieve_code(enhanced_query, k, filters)
        elif strategy == "api_search":
            return self._retrieve_api_docs(enhanced_query, k, filters)
        elif strategy == "hybrid":
            return self._retrieve_hybrid(enhanced_query, k, filters)
        else:
            return self._retrieve_general(enhanced_query, k, filters)
    
    def _enhance_query(self, query: str) -> str:
        """Enhance query with code-specific terms"""
        enhancements = []
        
        # Add programming language context
        languages = ['python', 'javascript', 'java', 'typescript', 'c++', 'go']
        for lang in languages:
            if lang.lower() in query.lower():
                enhancements.append(f"{lang} programming language")
        
        # Add common programming terms
        code_terms = {
            'function': ['function', 'method', 'def', 'func'],
            'class': ['class', 'object', 'struct', 'type'],
            'api': ['API', 'interface', 'endpoint', 'service'],
            'error': ['error', 'exception', 'bug', 'issue'],
            'import': ['import', 'include', 'require', 'using']
        }
        
        query_lower = query.lower()
        for term, synonyms in code_terms.items():
            if term in query_lower:
                enhancements.extend(synonyms)
        
        enhanced = query + " " + " ".join(enhancements)
        return enhanced
    
    def _determine_strategy(self, query: str) -> str:
        """Determine the best retrieval strategy based on query"""
        query_lower = query.lower()
        
        # Check for code-specific queries
        code_indicators = ['function', 'class', 'method', 'implement', 'code', 'example']
        if any(indicator in query_lower for indicator in code_indicators):
            return "code_search"
        
        # Check for API-specific queries
        api_indicators = ['api', 'endpoint', 'request', 'response', 'parameter']
        if any(indicator in query_lower for indicator in api_indicators):
            return "api_search"
        
        # Check for documentation queries
        doc_indicators = ['how to', 'what is', 'explain', 'documentation', 'guide']
        if any(indicator in query_lower for indicator in doc_indicators):
            return "hybrid"
        
        return "general"
    
    def _retrieve_code(self, query: str, k: int, filters: Optional[Dict]) -> List[Document]:
        """Retrieve code-specific documents"""
        # Add code-specific filters
        code_filters = filters or {}
        code_filters['chunk_type'] = 'code'
        
        # Perform retrieval
        results = self.vector_store.similarity_search_with_score(query, k=k*2, filter=code_filters)
        
        # Re-rank based on code relevance
        ranked_results = self._rerank_code_results(results, query)
        
        return [doc for doc, _ in ranked_results[:k]]
    
    def _retrieve_api_docs(self, query: str, k: int, filters: Optional[Dict]) -> List[Document]:
        """Retrieve API documentation"""
        # Search for API patterns in query
        api_patterns = re.findall(r'\b(\w+\.\w+|\w+\(\))\b', query)
        
        results = []
        for pattern in api_patterns:
            pattern_results = self.vector_store.similarity_search(pattern, k=k//2)
            results.extend(pattern_results)
        
        # Also do general search
        general_results = self.vector_store.similarity_search(query, k=k)
        results.extend(general_results)
        
        # Deduplicate and return top k
        seen = set()
        unique_results = []
        for doc in results:
            doc_id = f"{doc.metadata.get('file_path', '')}_{doc.metadata.get('chunk_index', 0)}"
            if doc_id not in seen:
                seen.add(doc_id)
                unique_results.append(doc)
                if len(unique_results) >= k:
                    break
        
        return unique_results
    
    def _retrieve_hybrid(self, query: str, k: int, filters: Optional[Dict]) -> List[Document]:
        """Hybrid retrieval combining code and documentation"""
        # Get code results
        code_results = self.vector_store.similarity_search(
            query, k=k//2, filter={'chunk_type': 'code'} if not filters else {**filters, 'chunk_type': 'code'}
        )
        
        # Get documentation results
        doc_results = self.vector_store.similarity_search(
            query, k=k//2, filter={'chunk_type': 'documentation'} if not filters else {**filters, 'chunk_type': 'documentation'}
        )
        
        # Combine and interleave results
        combined = []
        for i in range(max(len(code_results), len(doc_results))):
            if i < len(code_results):
                combined.append(code_results[i])
            if i < len(doc_results):
                combined.append(doc_results[i])
            if len(combined) >= k:
                break
        
        return combined[:k]
    
    def _retrieve_general(self, query: str, k: int, filters: Optional[Dict]) -> List[Document]:
        """General retrieval without specific strategy"""
        return self.vector_store.similarity_search(query, k=k, filter=filters)
    
    def _rerank_code_results(self, results: List[tuple], query: str) -> List[tuple]:
        """Re-rank code results based on relevance signals"""
        scored_results = []
        
        for doc, base_score in results:
            # Calculate additional relevance signals
            bonus_score = 0
            
            # Check for function/class names in query
            if 'function_name' in doc.metadata:
                if doc.metadata['function_name'].lower() in query.lower():
                    bonus_score += 0.2
            
            if 'class_name' in doc.metadata:
                if doc.metadata['class_name'].lower() in query.lower():
                    bonus_score += 0.2
            
            # Check for matching programming language
            if 'language' in doc.metadata:
                if doc.metadata['language'].lower() in query.lower():
                    bonus_score += 0.1
            
            # Check for docstrings
            if doc.metadata.get('docstring'):
                bonus_score += 0.05
            
            # Calculate final score
            final_score = base_score + bonus_score
            scored_results.append((doc, final_score))
        
        # Sort by final score
        scored_results.sort(key=lambda x: x[1], reverse=True)
        return scored_results
    
    def get_context_window(self, document: Document, window_size: int = 1) -> List[Document]:
        """Get surrounding chunks for more context"""
        file_path = document.metadata.get('file_path')
        chunk_index = document.metadata.get('chunk_index', 0)
        
        if not file_path:
            return [document]
        
        # Get surrounding chunks
        context_docs = []
        for i in range(max(0, chunk_index - window_size), chunk_index + window_size + 1):
            filter_dict = {
                'file_path': file_path,
                'chunk_index': i
            }
            
            chunks = self.vector_store.similarity_search("", k=1, filter=filter_dict)
            if chunks:
                context_docs.extend(chunks)
        
        return context_docs if context_docs else [document]