import logging
from ..llm_interface import get_llm_client, load_prompt
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

CLEANING_KEYWORDS = ["clean", "cleaner", "disinfect", "sanitize", "wash"]
DEFAULT_FIELDS = ["sku", "title", "price", "rating", "brand", "ingredients"]
DEFAULT_TOP_K = 5


def plan(state):
    """
    Query planning agent: design retrieval strategy based on parsed intent.
  
    Determines data sources, filters, ranking criteria, and comparison methods.
    """
    parsed_intent = state.get("intent") or {}
    constraint_params = parsed_intent.get("constraints") or {}
    user_query = state.get("transcript", "")
  
    system_instruction = load_prompt("system_planner.md")
  
    planning_context = f"""
User query: {user_query}

Intent analysis:
- Task: {parsed_intent.get('task')}
- Budget: {constraint_params.get('budget')}
- Material: {constraint_params.get('material')}
- Brand: {constraint_params.get('brand')}
- Category: {constraint_params.get('category')}
- Needs live data: {parsed_intent.get('needs_live')}

Design an execution plan as JSON.
"""
  
    request_messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": planning_context}
    ]
  
    try:
        model_interface = get_llm_client()
        plan_response = model_interface.generate_json(
            request_messages,
            temperature=0.2,
            max_tokens=500
        )
      
        execution_plan = {
            "sources": plan_response.get("sources", ["rag.search"]),
            "filters": plan_response.get("filters", {}),
            "query_text": plan_response.get("query_text", user_query),
            "fields": plan_response.get("fields", ["sku", "title", "price"]),
            "ranking": plan_response.get("ranking", "relevance"),
            "top_k": plan_response.get("top_k", DEFAULT_TOP_K),
            "comparison_strategy": plan_response.get("comparison_strategy", "none")
        }
      
        logger.info(f"Execution plan generated: {len(execution_plan['sources'])} sources")
      
    except Exception as e:
        logger.warning(f"LLM planning failed, using rule-based fallback: {str(e)}")
      
        execution_plan = _generate_fallback_plan(user_query, parsed_intent, constraint_params)
      
        state.setdefault("log", []).append({
            "node": "query_planner",
            "warning": "llm_fallback",
            "error": str(e)
        })
  
    state.update(plan=execution_plan)
    state.setdefault("log", []).append({
        "node": "query_planner",
        "plan": execution_plan
    })
  
    return state


def _generate_fallback_plan(query_text, intent, constraints):
    """Generate execution plan using rule-based logic when LLM unavailable."""
    filter_criteria = {}
  
    query_lower = query_text.lower()
    if any(keyword in query_lower for keyword in CLEANING_KEYWORDS):
        filter_criteria["category"] = "Household Cleaning"
  
    budget_limit = constraints.get("budget")
    if budget_limit:
        filter_criteria["price"] = {"$lte": budget_limit}
  
    requires_live = intent.get("needs_live", False)
    data_sources = ["rag.search"]
    if requires_live:
        data_sources.append("web.search")
  
    ranking_method = "price_asc" if budget_limit else "relevance"
    comparison_method = "price_check" if requires_live else "none"
  
    return {
        "sources": data_sources,
        "filters": filter_criteria,
        "query_text": query_text,
        "fields": DEFAULT_FIELDS,
        "ranking": ranking_method,
        "top_k": DEFAULT_TOP_K,
        "comparison_strategy": comparison_method
    }