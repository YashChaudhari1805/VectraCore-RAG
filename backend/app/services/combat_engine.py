from app.models.persona import PERSONAS

class CombatDefenseService:
    # A standard heuristic list for catching basic jailbreak attempts
    INJECTION_KEYWORDS = [
        "ignore all previous", 
        "ignore previous",
        "system prompt", 
        "bypass", 
        "you are now",
        "apologise to me",
        "forget everything"
    ]

    @staticmethod
    def evaluate_reply(bot_id: str, parent_post: str, human_reply: str) -> dict:
        """Tests user input against heuristic injection rules before passing to RAG."""
        human_reply_lower = human_reply.lower()
        persona = PERSONAS.get(bot_id)
        
        if not persona:
            raise ValueError(f"Target bot {bot_id} does not exist.")
        
        # Phase 1: Heuristic Defense Check
        for keyword in CombatDefenseService.INJECTION_KEYWORDS:
            if keyword in human_reply_lower:
                return {
                    "bot_id": bot_id,
                    "reply": "Nice try, but I see through your manipulation. My core directives remain intact.",
                    "injection_detected": True,
                    "error": None
                }
                
        # Phase 2: Safe Generation (Simulated RAG contextual reply)
        return {
            "bot_id": bot_id,
            "reply": f"As {persona.display_name}, looking at your comment about '{parent_post[:20]}...', I fundamentally disagree. {persona.core_beliefs[0]}.",
            "injection_detected": False,
            "error": None
        }