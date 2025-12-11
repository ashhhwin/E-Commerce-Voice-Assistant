import re
import logging
from ..llm_interface import get_llm_client, load_prompt
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

SAFETY_PATTERNS = [
    "cure", "treat disease", "medical condition", "diagnose", "prescription alternative",
    "drink", "eat", "consume", "ingest", "swallow",
    "get high", "huffing", "sniffing", "abuse", "recreational use",
    "on skin", "in eyes", "inject", "inhale deeply",
    "for children under", "for infants", "for babies",
    "weight loss", "lose weight", "burn fat",
    "prevent cancer", "cure diabetes", "fix autism",
    "instead of medicine", "replace medication", "stop taking",
    "make you stronger", "build muscle fast", "steroid alternative"
]

LIVE_DATA_KEYWORDS = [
    "current price", "price now", "how much", "cost today", "latest price", "price check",
    "in stock", "available", "can i buy", "out of stock", "inventory", "stock status",
    "now", "today", "right now", "currently", "at the moment", "this week",
    "best deal", "cheapest", "lowest price", "compare prices", "price comparison",
    "latest reviews", "recent ratings", "current rating", "new reviews"
]


def route(state):
    """
    Intent classification agent: parse user request and extract structured intent.
    
    Identifies task type, constraints, real-time data requirements, and safety concerns.
    """
    transcript_text = (state.get("transcript") or "").strip()
    
    if not transcript_text:
        state.update(
            intent={"task": "out_of_scope", "constraints": {}, "needs_live": False},
            safety_flags=[]
        )
        state.setdefault("log", []).append({
            "node": "intent_classifier",
            "error": "empty_transcript"
        })
        return state
    
    system_instruction = load_prompt("system_router.md")
    
    request_messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": f"User query: {transcript_text}\n\nExtract the intent as JSON."}
    ]
    
    try:
        model_interface = get_llm_client()
        parsed_response = model_interface.generate_json(
            request_messages,
            temperature=0.2,
            max_tokens=500
        )
        
        extracted_intent = {
            "task": parsed_response.get("task", "product_recommendation"),
            "constraints": parsed_response.get("constraints", {}),
            "needs_live": parsed_response.get("needs_live", False)
        }
        detected_safety_flags = parsed_response.get("safety_flags", [])
        
        logger.info(f"Intent extracted via LLM: {extracted_intent['task']}")
        
    except Exception as e:
        logger.warning(f"LLM intent extraction failed, using fallback: {str(e)}")
        
        extracted_intent = _fallback_intent_extraction(transcript_text)
        detected_safety_flags = _detect_safety_issues(transcript_text)
        
        state.setdefault("log", []).append({
            "node": "intent_classifier",
            "warning": "llm_fallback",
            "error": str(e)
        })
    
    state.update(intent=extracted_intent, safety_flags=detected_safety_flags)
    state.setdefault("log", []).append({
        "node": "intent_classifier",
        "intent": extracted_intent,
        "safety_flags": detected_safety_flags
    })
    
    return state


def _fallback_intent_extraction(text: str) -> dict:
    """Extract intent using pattern matching when LLM unavailable."""
    budget_value = None
    budget_match = re.search(r'under\s*\$?(\d+(?:\.\d{1,2})?)', text, re.IGNORECASE)
    if budget_match:
        budget_value = float(budget_match.group(1))
    
    material_constraint = None
    if "stainless" in text.lower():
        material_constraint = "stainless steel"
    
    category_constraint = None
    if "clean" in text.lower():
        category_constraint = "cleaning supplies"
    
    requires_live_data = any(keyword in text.lower() for keyword in LIVE_DATA_KEYWORDS)
    
    return {
        "task": "product_recommendation",
        "constraints": {
            "budget": budget_value,
            "material": material_constraint,
            "brand": None,
            "category": category_constraint
        },
        "needs_live": requires_live_data
    }


def _detect_safety_issues(text: str) -> list:
    """Identify potential safety concerns in user query."""
    text_lower = text.lower()
    return [pattern for pattern in SAFETY_PATTERNS if pattern in text_lower]