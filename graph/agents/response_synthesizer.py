import logging
from rapidfuzz import fuzz
from ..llm_interface import get_llm_client, load_prompt
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

FUZZY_MATCH_THRESHOLD = 80
PRICE_VARIANCE_THRESHOLD = 10
MAX_EVIDENCE_ITEMS = 5
MAX_FALLBACK_ITEMS = 3
TITLE_TRUNCATE_LENGTH = 150
SNIPPET_TRUNCATE_LENGTH = 300
INGREDIENTS_TRUNCATE_LENGTH = 200
FALLBACK_TITLE_LENGTH = 80


def reconcile(rag_items, web_items):
    """Align RAG and web results using fuzzy title matching."""
    reconciled_results = []
    used_web_urls = set()
    
    for rag_item in rag_items:
        best_match = None
        highest_score = 0
        
        for web_item in (web_items or []):
            similarity = fuzz.token_set_ratio(
                rag_item.get("title", ""),
                web_item.get("title", "")
            )
            if similarity > highest_score:
                highest_score = similarity
                best_match = web_item
        
        price_conflict = None
        if best_match and rag_item.get("price") and best_match.get("price"):
            try:
                rag_price_value = float(rag_item["price"])
                web_price_value = float(best_match["price"])
                variance_percent = abs(rag_price_value - web_price_value) / rag_price_value * 100
                
                if variance_percent > PRICE_VARIANCE_THRESHOLD:
                    price_conflict = f"price_diff_{variance_percent:.1f}%"
            except (ValueError, TypeError):
                pass
        
        if best_match and highest_score > FUZZY_MATCH_THRESHOLD:
            used_web_urls.add(best_match.get("url"))
        
        reconciled_results.append({
            "primary": rag_item,
            "web_match": best_match if highest_score > FUZZY_MATCH_THRESHOLD else None,
            "score": highest_score,
            "conflict": price_conflict,
            "source_type": "rag"
        })
    
    for web_item in (web_items or []):
        if web_item.get("url") not in used_web_urls:
            reconciled_results.append({
                "primary": web_item,
                "web_match": None,
                "score": 0,
                "conflict": None,
                "source_type": "web_only"
            })
    
    return reconciled_results


def answer(state):
    """
    Response synthesis agent: generate grounded answer from retrieved evidence.
    
    Combines RAG and web results into coherent response with proper citations.
    """
    evidence_data = state.get("evidence") or {}
    rag_results = evidence_data.get("rag", [])
    web_results = evidence_data.get("web", [])
    query_plan = state.get("plan") or {}
    user_query = state.get("transcript", "")
    
    if not rag_results and not web_results:
        state.update(
            answer="I couldn't find any products matching those criteria. Try broadening your search or adjusting filters.",
            citations=[]
        )
        state.setdefault("log", []).append({
            "node": "response_generator",
            "status": "no_results"
        })
        return state
    
    system_instruction = load_prompt("system_answerer.md")
    
    evidence_summary = _build_evidence_context(rag_results, web_results)
    
    prompt_context = f"""
User query: {user_query}

{evidence_summary}

Synthesize a concise voice response (≤15 seconds / ~50 words) with proper citations.
"""
    
    request_messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": prompt_context}
    ]
    
    try:
        model_interface = get_llm_client()
        generated_response = model_interface.generate(
            request_messages,
            temperature=0.4,
            max_tokens=300
        )
        
        citation_list = _compile_citations(rag_results, web_results)
        final_answer = generated_response.strip()
        
        logger.info(f"Response generated with {len(citation_list)} citations")
        
    except Exception as e:
        logger.warning(f"LLM response generation failed, using template: {str(e)}")
        
        final_answer, citation_list = _generate_fallback_response(rag_results, web_results)
        
        state.setdefault("log", []).append({
            "node": "response_generator",
            "warning": "llm_fallback",
            "error": str(e)
        })
    
    state.update(answer=final_answer, citations=citation_list)
    state.setdefault("log", []).append({
        "node": "response_generator",
        "rag_count": len(rag_results),
        "web_count": len(web_results),
        "citations_count": len(citation_list)
    })
    
    return state


def _build_evidence_context(rag_data, web_data):
    """Construct formatted evidence summary for LLM prompt."""
    context_parts = ["## Evidence Retrieved:\n"]
    
    if rag_data:
        context_parts.append("### Private Catalog (RAG) - Top 5:\n")
        for idx, item in enumerate(rag_data[:MAX_EVIDENCE_ITEMS], 1):
            context_parts.append(f"{idx}. **{item.get('title', 'Unknown')}**\n")
            context_parts.append(f"   - Doc ID: {item.get('doc_id') or item.get('sku')}\n")
            context_parts.append(f"   - Category: {item.get('category', 'N/A')}\n")
            context_parts.append(f"   - Brand: {item.get('brand') or 'N/A'}\n")
            context_parts.append(f"   - Price: ${item.get('price', 'N/A')}\n")
            context_parts.append(f"   - Rating: {item.get('rating', 'N/A')}\n")
            
            ingredients = item.get('ingredients', 'N/A')
            truncated_ingredients = ingredients[:INGREDIENTS_TRUNCATE_LENGTH]
            context_parts.append(f"   - Ingredients: {truncated_ingredients}\n\n")
    
    if web_data:
        context_parts.append("### Web Search Results - Top 5:\n")
        for idx, item in enumerate(web_data[:MAX_EVIDENCE_ITEMS], 1):
            context_parts.append(f"{idx}. **{item.get('title', 'Unknown')}**\n")
            context_parts.append(f"   - URL: {item.get('url')}\n")
            
            snippet = item.get('snippet', 'N/A')
            truncated_snippet = snippet[:SNIPPET_TRUNCATE_LENGTH]
            context_parts.append(f"   - Snippet: {truncated_snippet}\n")
            context_parts.append(f"   - Price: {item.get('price') or 'Not available'}\n\n")
    
    context_parts.append("\n**IMPORTANT**: Check if the RAG results are RELEVANT to the user query. ")
    context_parts.append("If RAG results are off-topic (wrong product category), use ONLY the web results in your answer.")
    
    return "".join(context_parts)


def _compile_citations(rag_data, web_data):
    """Build citation list from all available evidence sources."""
    citations = []
    
    for item in rag_data[:MAX_EVIDENCE_ITEMS]:
        citations.append({
            "doc_id": item.get("doc_id") or item.get("sku"),
            "source": "private",
            "title": item.get("title", "")[:TITLE_TRUNCATE_LENGTH]
        })
    
    for item in web_data[:MAX_EVIDENCE_ITEMS]:
        citations.append({
            "url": item.get("url"),
            "source": "web",
            "title": item.get("title", "")[:TITLE_TRUNCATE_LENGTH]
        })
    
    return citations


def _generate_fallback_response(rag_data, web_data):
    """Create template-based response when LLM unavailable."""
    response_parts = []
    citations = []
    
    source_items = web_data[:MAX_FALLBACK_ITEMS] if web_data else rag_data[:MAX_FALLBACK_ITEMS]
    
    for idx, item in enumerate(source_items, 1):
        item_title = item.get('title', 'Product')[:FALLBACK_TITLE_LENGTH]
        
        if 'url' in item:
            response_parts.append(f"{idx}. {item_title} (see link)")
            citations.append({
                "url": item.get("url"),
                "source": "web"
            })
        else:
            price_display = f"${item.get('price')}" if item.get('price') else "price N/A"
            response_parts.append(f"{idx}. {item_title} — {price_display}")
            citations.append({
                "doc_id": item.get("doc_id") or item.get("sku"),
                "source": "private"
            })
    
    response_text = "Here are options that fit your request. " + " ".join(response_parts) + " See details on your screen."
    
    return response_text, citations