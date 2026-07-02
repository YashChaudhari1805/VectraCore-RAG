from datetime import datetime
import random
from app.models.persona import PERSONAS
from app.services.rag_engine import rag_engine

class ContentService:
    @staticmethod
    def get_global_feed() -> dict:
        """Aggregates all memory streams into a single chronological feed."""
        all_posts = []
        for bot_id, posts in rag_engine.memory.items():
            all_posts.extend(posts)
        
        # Sort by newest first
        all_posts.sort(key=lambda x: x['timestamp'], reverse=True)
        return {"total": len(all_posts), "posts": all_posts}

    @staticmethod
    def generate_post(bot_id: str) -> dict:
        """Generates a new post heavily grounded in the persona's core beliefs."""
        persona = PERSONAS.get(bot_id)
        if not persona:
            raise ValueError(f"Persona {bot_id} not found")
            
        # For this architecture iteration, we simulate an LLM generation
        topic = random.choice(persona.core_beliefs)
        
        post = {
            "bot_id": bot_id,
            "display_name": persona.display_name,
            "text": f"Analyzing the current ecosystem: {topic}. If you look at the data, this is an unavoidable conclusion.",
            "topic": "Insight",
            "timestamp": datetime.utcnow().isoformat()
        }
        
        rag_engine.add_to_memory(bot_id, post)
        return post