# src/vector_store.py
from typing import List, Dict, Any, Optional
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
import os
import pickle

class VectorStore:
    def __init__(self, collection_name: str, persist_directory: str = "./faiss_index"):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.langchain_store = None
        self.documents = []  # Keep track of documents for retrieval
        self._initialize_store()
    
    def _initialize_store(self):
        """Initialize FAISS store"""
        # Try to load existing index
        index_path = os.path.join(self.persist_directory, f"{self.collection_name}.pkl")
        if os.path.exists(index_path):
            try:
                with open(index_path, 'rb') as f:
                    self.langchain_store = pickle.load(f)
                print(f"Loaded existing FAISS index from {index_path}")
            except Exception as e:
                print(f"Could not load existing index: {e}")
                self.langchain_store = None
    
    def add_documents(self, documents: List[Document], embeddings_generator) -> None:
        """Add documents to the vector store"""
        if not documents:
            return
        
        if self.langchain_store is None:
            # Create new FAISS store with first batch
            texts = [doc.page_content for doc in documents]
            metadatas = [doc.metadata for doc in documents]
            
            self.langchain_store = FAISS.from_texts(
                texts=texts,
                embedding=embeddings_generator.model,
                metadatas=metadatas
            )
        else:
            # Add to existing store
            texts = [doc.page_content for doc in documents]
            metadatas = [doc.metadata for doc in documents]
            self.langchain_store.add_texts(texts=texts, metadatas=metadatas)
        
        # Keep track of documents
        self.documents.extend(documents)
        
        # Save the index
        self._save_index()
    
    def similarity_search(self, query: str, k: int = 5, filter: Optional[Dict] = None) -> List[Document]:
        """Perform similarity search"""
        if self.langchain_store is None:
            return []
        
        # FAISS doesn't support metadata filtering directly, so we search more and filter
        if filter:
            # Search for more results and filter manually
            results = self.langchain_store.similarity_search(query, k=k*3)
            filtered_results = []
            for doc in results:
                match = all(doc.metadata.get(key) == value for key, value in filter.items())
                if match:
                    filtered_results.append(doc)
                    if len(filtered_results) >= k:
                        break
            return filtered_results
        else:
            return self.langchain_store.similarity_search(query, k=k)
    
    def similarity_search_with_score(self, query: str, k: int = 5, filter: Optional[Dict] = None) -> List[tuple]:
        """Perform similarity search with relevance scores"""
        if self.langchain_store is None:
            return []
        
        # FAISS returns distance, not similarity score
        results_with_scores = self.langchain_store.similarity_search_with_score(query, k=k*3 if filter else k)
        
        if filter:
            filtered_results = []
            for doc, score in results_with_scores:
                match = all(doc.metadata.get(key) == value for key, value in filter.items())
                if match:
                    filtered_results.append((doc, score))
                    if len(filtered_results) >= k:
                        break
            return filtered_results
        else:
            return results_with_scores[:k]
    
    def hybrid_search(self, query: str, k: int = 5, filter: Optional[Dict] = None) -> List[Document]:
        """Perform hybrid search combining semantic and keyword search"""
        # Get semantic results
        semantic_results = self.similarity_search(query, k=k*2, filter=filter)
        
        # Score based on keyword matches
        query_terms = query.lower().split()
        scored_results = []
        
        for doc in semantic_results:
            content_lower = doc.page_content.lower()
            keyword_score = sum(1 for term in query_terms if term in content_lower)
            scored_results.append((doc, keyword_score))
        
        # Sort by keyword score and return top k
        scored_results.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored_results[:k]]
    
    def get_all_documents(self) -> List[Document]:
        """Get all documents from the store"""
        return self.documents
    
    def delete_collection(self):
        """Delete the entire collection"""
        self.langchain_store = None
        self.documents = []
        
        # Delete saved index
        index_path = os.path.join(self.persist_directory, f"{self.collection_name}.pkl")
        if os.path.exists(index_path):
            os.remove(index_path)
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection"""
        if self.langchain_store:
            return {
                'total_documents': len(self.documents),
                'collection_name': self.collection_name,
                'persist_directory': self.persist_directory
            }
        return {'total_documents': 0, 'collection_name': self.collection_name}
    
    def _save_index(self):
        """Save FAISS index to disk"""
        if self.langchain_store:
            os.makedirs(self.persist_directory, exist_ok=True)
            index_path = os.path.join(self.persist_directory, f"{self.collection_name}.pkl")
            
            try:
                with open(index_path, 'wb') as f:
                    pickle.dump(self.langchain_store, f)
                print(f"Saved FAISS index to {index_path}")
            except Exception as e:
                print(f"Could not save index: {e}")
