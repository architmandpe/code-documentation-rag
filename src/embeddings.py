from typing import List, Optional
from langchain_openai import OpenAIEmbeddings
from sentence_transformers import SentenceTransformer
import numpy as np

class EmbeddingGenerator:
    def __init__(self, model_type: str = "openai", model_name: Optional[str] = None):
        self.model_type = model_type
        self.model_name = model_name
        self.model = self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the embedding model"""
        if self.model_type == "openai":
            from config.settings import settings
            return OpenAIEmbeddings(
                model=self.model_name or settings.EMBEDDING_MODEL,
                openai_api_key=settings.OPENAI_API_KEY
            )
        elif self.model_type == "sentence-transformers":
            model_name = self.model_name or "sentence-transformers/all-mpnet-base-v2"
            return SentenceTransformer(model_name)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts"""
        if self.model_type == "openai":
            return self.model.embed_documents(texts)
        elif self.model_type == "sentence-transformers":
            embeddings = self.model.encode(texts, convert_to_tensor=False)
            return embeddings.tolist()
    
    def generate_query_embedding(self, query: str) -> List[float]:
        """Generate embedding for a query"""
        if self.model_type == "openai":
            return self.model.embed_query(query)
        elif self.model_type == "sentence-transformers":
            embedding = self.model.encode(query, convert_to_tensor=False)
            return embedding.tolist()
    
    def compute_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Compute cosine similarity between two embeddings"""
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        cosine_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        return float(cosine_sim)