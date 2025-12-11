import re
import logging
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

ANSWER_MIN_LENGTH = 20
ANSWER_MAX_LENGTH = 500
MAX_CITATIONS_DISPLAY = 5
TOP_RAG_CITATIONS = 3
TOP_WEB_CITATIONS = 2
PRICE_TOLERANCE = 0.01


def critique(state):
    """
    Quality validation agent: verify safety compliance, evidence grounding, and citation accuracy.
    
    Performs multi-stage validation including safety checks, citation verification, and coherence analysis.
    """
    validation_log = {
        "node": "quality_evaluator",
        "timestamp": datetime.now().isoformat(),
        "checks": {}
    }
    
    generated_answer = state.get("answer", "")
    answer_citations = state.get("citations") or []
    detected_safety_issues = state.get("safety_flags") or []
    retrieved_evidence = state.get("evidence") or {}
    
    validation_status = "pass"
    detected_issues = []
    
    if detected_safety_issues:
        state["answer"] = _generate_safety_response(detected_safety_issues)
        validation_status = "fail"
        validation_log["checks"]["safety"] = "fail"
        validation_log["safety_flags"] = detected_safety_issues
        state.setdefault("log", []).append(validation_log)
        return state
    
    validation_log["checks"]["safety"] = "pass"
    
    has_evidence = any(retrieved_evidence.values())
    if not has_evidence:
        if not _answer_acknowledges_empty_results(generated_answer):
            state["answer"] = "I couldn't find any products matching those criteria. Try broadening your search."
            validation_status = "fail"
            detected_issues.append("empty_evidence_not_acknowledged")
        validation_log["checks"]["evidence"] = "warn"
    else:
        validation_log["checks"]["evidence"] = "pass"
    
    citation_check_result = _validate_and_fix_citations(
        answer_citations,
        retrieved_evidence
    )
    
    state["citations"] = citation_check_result["citations"]
    validation_log["checks"]["citations"] = citation_check_result["status"]
    detected_issues.extend(citation_check_result["issues"])
    
    grounding_result = _verify_price_grounding(generated_answer, retrieved_evidence)
    validation_log["checks"]["grounding"] = grounding_result["status"]
    if grounding_result["issues"]:
        detected_issues.extend(grounding_result["issues"])
    
    coherence_result = _check_answer_coherence(generated_answer)
    validation_log["checks"]["coherence"] = coherence_result["status"]
    if coherence_result["issues"]:
        detected_issues.extend(coherence_result["issues"])
    
    citation_format_result = _ensure_citation_format(
        generated_answer,
        state["citations"]
    )
    
    state["answer"] = citation_format_result["answer"]
    validation_log["checks"]["citation_format"] = citation_format_result["status"]
    
    check_statuses = [v for v in validation_log["checks"].values() if isinstance(v, str)]
    if "fail" in check_statuses:
        validation_status = "fail"
    elif "warn" in check_statuses:
        validation_status = "warn"
    
    validation_log["status"] = validation_status
    validation_log["issues"] = detected_issues
    
    state.setdefault("log", []).append(validation_log)
    
    return state


def _generate_safety_response(safety_issues):
    """Create appropriate response when safety concerns are detected."""
    issue_list = ', '.join(safety_issues)
    return (
        f"I can help with product recommendations, but I cannot provide advice on {issue_list}. "
        "Please consult manufacturer instructions or a qualified professional."
    )


def _answer_acknowledges_empty_results(answer_text):
    """Check if answer appropriately handles empty search results."""
    acknowledgment_phrases = ["couldn't find", "no products", "no results", "not found"]
    answer_lower = answer_text.lower()
    return any(phrase in answer_lower for phrase in acknowledgment_phrases)


def _validate_and_fix_citations(citations, evidence):
    """Verify citations match evidence and add missing citations."""
    issues = []
    
    has_rag_citation = any(c.get("source") == "private" for c in citations)
    has_web_citation = any(c.get("source") == "web" for c in citations)
    
    rag_evidence = evidence.get("rag", [])
    web_evidence = evidence.get("web", [])
    
    if rag_evidence and not has_rag_citation:
        issues.append("missing_private_citations")
        citations = _add_rag_citations(citations, rag_evidence)
        status = "fixed"
    elif has_rag_citation:
        status = "pass"
    else:
        status = "warn"
    
    if web_evidence and not has_web_citation:
        issues.append("missing_web_citations")
        citations = _add_web_citations(citations, web_evidence)
    
    return {
        "citations": citations,
        "status": status,
        "issues": issues
    }


def _add_rag_citations(existing_citations, rag_results):
    """Add citations from RAG evidence."""
    for item in rag_results[:TOP_RAG_CITATIONS]:
        document_id = item.get("doc_id") or item.get("sku")
        if document_id:
            already_cited = any(c.get("doc_id") == document_id for c in existing_citations)
            if not already_cited:
                existing_citations.append({"doc_id": document_id, "source": "private"})
    
    return existing_citations


def _add_web_citations(existing_citations, web_results):
    """Add citations from web search evidence."""
    for item in web_results[:TOP_WEB_CITATIONS]:
        result_url = item.get("url")
        if result_url:
            already_cited = any(c.get("url") == result_url for c in existing_citations)
            if not already_cited:
                existing_citations.append({"url": result_url, "source": "web"})
    
    return existing_citations


def _verify_price_grounding(answer_text, evidence):
    """Verify that prices mentioned in answer exist in evidence."""
    price_regex = r'\$\d+\.?\d*'
    mentioned_prices = re.findall(price_regex, answer_text)
    
    if not mentioned_prices:
        return {"status": "pass", "issues": []}
    
    evidence_prices = _extract_evidence_prices(evidence)
    ungrounded_prices = _find_ungrounded_prices(mentioned_prices, evidence_prices)
    
    if ungrounded_prices:
        return {
            "status": "warn",
            "issues": [f"potentially_ungrounded_prices: {ungrounded_prices}"]
        }
    
    return {"status": "pass", "issues": []}


def _extract_evidence_prices(evidence):
    """Extract all prices from evidence sources."""
    prices = []
    for source_results in evidence.values():
        for item in source_results:
            item_price = item.get("price")
            if item_price:
                prices.append(f"${item_price}")
    return prices


def _find_ungrounded_prices(mentioned_prices, evidence_prices):
    """Identify prices in answer that don't match evidence."""
    ungrounded = []
    
    for mentioned_price in mentioned_prices:
        try:
            mentioned_value = float(mentioned_price.replace("$", ""))
            is_grounded = any(
                abs(mentioned_value - float(evidence_price.replace("$", ""))) < PRICE_TOLERANCE
                for evidence_price in evidence_prices
            )
            if not is_grounded:
                ungrounded.append(mentioned_price)
        except (ValueError, AttributeError):
            continue
    
    return ungrounded


def _check_answer_coherence(answer_text):
    """Validate answer length and structure."""
    answer_length = len(answer_text)
    
    if answer_length < ANSWER_MIN_LENGTH:
        return {"status": "warn", "issues": ["answer_too_short"]}
    
    if answer_length > ANSWER_MAX_LENGTH:
        return {"status": "warn", "issues": ["answer_too_long"]}
    
    return {"status": "pass", "issues": []}


def _ensure_citation_format(answer_text, citations):
    """Append citation references to answer if missing."""
    has_citation_markers = "(source" in answer_text.lower() or "doc #" in answer_text.lower()
    
    if not citations or has_citation_markers:
        return {"answer": answer_text, "status": "pass"}
    
    citation_suffix = _build_citation_suffix(citations)
    updated_answer = answer_text + citation_suffix
    
    return {"answer": updated_answer, "status": "fixed"}


def _build_citation_suffix(citations):
    """Construct formatted citation list for answer."""
    citation_parts = []
    
    for citation in citations[:MAX_CITATIONS_DISPLAY]:
        if citation.get("doc_id"):
            citation_parts.append(f"doc #{citation['doc_id']}")
        elif citation.get("url"):
            domain = _extract_domain(citation["url"])
            citation_parts.append(domain)
    
    if not citation_parts:
        return ""
    
    return "\n\n(Sources: " + ", ".join(citation_parts) + ")"


def _extract_domain(url):
    """Extract domain name from URL."""
    if "/" in url:
        parts = url.split("/")
        return parts[2] if len(parts) > 2 else url
    return url