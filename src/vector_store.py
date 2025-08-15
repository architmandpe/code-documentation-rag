from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
import uuid

class VectorStore:
    def __init__(self, collection_name: str, persist_directory: str = "./chroma_db"):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self.langchain_store = None
        self._initialize_store()
    
    def _initialize_store(self):
        """Initialize ChromaDB client and collection"""
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Create or get collection
        try:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        except:
            self.collection = self.client.get_collection(name=self.collection_name)
    
    def add_documents(self, documents: List[Document], embeddings_generator) -> None:
        """Add documents to the vector store"""
        if self.langchain_store is None:
            self.langchain_store = Chroma(
                collection_name=self.collection_name,
                embedding_function=embeddings_generator.model,
                persist_directory=self.persist_directory
            )
        
        # Add documents in batches
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]
            self.langchain_store.add_documents(batch)
    
    def similarity_search(self, query: str, k: int = 5, filter: Optional[Dict] = None) -> List[Document]:
        """Perform similarity search"""
        if self.langchain_store is None:
            return []
        
        if filter:
            return self.langchain_store.similarity_search(
                query, k=k, filter=filter
            )
        else:
            return self.langchain_store.similarity_search(query, k=k)
    
    def similarity_search_with_score(self, query: str, k: int = 5, filter: Optional[Dict] = None) -> List[tuple]:
        """Perform similarity search with relevance scores"""
        if self.langchain_store is None:
            return []
        
        return self.langchain_store.similarity_search_with_relevance_scores(
            query, k=k, filter=filter if filter else None
        )
    
    def hybrid_search(self, query: str, k: int = 5, filter: Optional[Dict] = None) -> List[Document]:
        """Perform hybrid search combining semantic and keyword search"""
        # Semantic search
        semantic_results = self.similarity_search(query, k=k*2, filter=filter)
        
        # Keyword search (simple implementation)
        keyword_results = []
        query_terms = query.lower().split()
        
        all_docs = self.get_all_documents()
        for doc in all_docs:
            content_lower = doc.page_content.lower()
            score = sum(1 for term in query_terms if term in content_lower)
            if score > 0:
                keyword_results.append((doc, score))
        
        # Sort keyword results by score
        keyword_results.sort(key=lambda x: x[1], reverse=True)
        keyword_docs = [doc for doc, _ in keyword_results[:k]]
        
        # Combine and deduplicate results
        combined = semantic_results + keyword_docs
        seen = set()
        unique_results = []
        
        for doc in combined:
            doc_id = doc.metadata.get('file_path', '') + str(doc.metadata.get('chunk_index', 0))
            if doc_id not in seen:
                seen.add(doc_id)
                unique_results.append(doc)
                if len(unique_results) >= k:
                    break
        
        return unique_results
    
    def get_all_documents(self) -> List[Document]:
        """Get all documents from the store"""
        if self.collection:
            results = self.collection.get()
            documents = []
            for i in range(len(results['ids'])):
                doc = Document(
                    page_content=results['documents'][i] if results['documents'] else "",
                    metadata=results['metadatas'][i] if results['metadatas'] else {}
                )
                documents.append(doc)
            return documents
        return []
    
    def delete_collection(self):
        """Delete the entire collection"""
        if self.client and self.collection_name:
            self.client.delete_collection(name=self.collection_name)
            self.collection = None
            self.langchain_store = None
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection"""
        if self.collection:
            count = self.collection.count()
            return {
                'total_documents': count,
                'collection_name': self.collection_name,
                'persist_directory': self.persist_directory
            }
        return {}