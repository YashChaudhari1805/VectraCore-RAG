import faiss
import numpy as np
from fastembed import TextEmbedding
from typing import List, Dict, Any
from app.core.config import get_settings
from app.models.persona import PERSONAS
import logging

logger = logging.getLogger(__name__)
settings = get_settings()

class RAGEngine:
    def __init__(self):
        # Load the local embedding model dynamically from settings
        logger.info(f"Loading embedding model: {settings.MODEL_NAME}")
        self.encoder = TextEmbedding(model_name=settings.MODEL_NAME)
        
        # Initialize FAISS index for Cosine Similarity (Inner Product on normalized vectors)
        self.router_index = faiss.IndexFlatIP(settings.EMBEDDING_DIMENSIONS)
        self.router_map: List[str] = [] 
        
        # In-memory data store (In a real enterprise app, this would be Postgres/Redis)
        self.memory: Dict[str, List[Dict[str, Any]]] = {bot_id: [] for bot_id in PERSONAS}
        
    def build_persona_index(self):
        """Embeds persona descriptions and core beliefs to establish the Semantic Router."""
        logger.info("Building persona FAISS index...")
        embeddings = []
        
        for bot_id, persona in PERSONAS.items():
            text_to_embed = f"{persona.description} {' '.join(persona.core_beliefs)}"
            emb = next(self.encoder.embed([text_to_embed]))
            embeddings.append(emb)
            self.router_map.append(bot_id)
            logger.debug(f"persona_embedded bot_id={bot_id}")
            
        if embeddings:
            vectors = np.array(embeddings).astype('float32')
            faiss.normalize_L2(vectors)
            self.router_index.add(vectors)
            logger.info(f"persona_index_ready dim={settings.EMBEDDING_DIMENSIONS} total_personas={len(PERSONAS)}")
            
    def semantic_route(self, text: str, threshold: float = 0.3) -> List[Dict[str, Any]]:
        """Routes a post to the most relevant personas based on semantic similarity."""
        query_vector = np.array(list(self.encoder.embed([text]))).astype('float32')
        faiss.normalize_L2(query_vector)
        
        k = len(self.router_map)
        if k == 0: return []
        
        similarities, indices = self.router_index.search(query_vector, k)
        
        results = []
        for i in range(k):
            sim = float(similarities[0][i])
            if sim >= threshold:
                results.append({
                    "bot_id": self.router_map[indices[0][i]], 
                    "similarity": sim
                })
                
        return sorted(results, key=lambda x: x["similarity"], reverse=True)
        
    def add_to_memory(self, bot_id: str, post: Dict[str, Any]):
        """Persists a generated post into the agent's memory stream."""
        if bot_id in self.memory:
            self.memory[bot_id].insert(0, post)
            
    def get_memory(self, bot_id: str) -> List[Dict[str, Any]]:
        return self.memory.get(bot_id, [])

# Export a singleton instance to be shared across the application
rag_engine = RAGEngine()