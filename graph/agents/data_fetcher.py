import os
import time
import logging
import httpx
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

RAG_SEARCH_TOOL = "rag.search"
WEB_SEARCH_TOOL = "web.search"
DEFAULT_RESULT_LIMIT = 5
WEB_RESULT_LIMIT = 5
REQUEST_TIMEOUT = 20


def retrieve(state):
    """
    Data retrieval agent: execute search operations based on query plan.
  
    Fetches evidence from configured data sources and aggregates results.
    """
    mcp_endpoint = os.getenv("MCP_BASE", "http://127.0.0.1:8000")
    query_plan = state.get("plan") or {}
    data_sources = query_plan.get("sources", [RAG_SEARCH_TOOL])
  
    collected_evidence = {}
    execution_log = []
  
    if RAG_SEARCH_TOOL in data_sources:
        rag_results, rag_metadata = _execute_rag_search(
            mcp_endpoint,
            query_plan,
            state.get("transcript", "")
        )
        collected_evidence["rag"] = rag_results
        execution_log.append(rag_metadata)
  
    if WEB_SEARCH_TOOL in data_sources:
        web_results, web_metadata = _execute_web_search(
            mcp_endpoint,
            query_plan,
            state.get("transcript", "")
        )
        collected_evidence["web"] = web_results
        execution_log.append(web_metadata)
  
    state.update(evidence=collected_evidence)
    state.setdefault("log", []).append({
        "node": "data_retriever",
        "tool_calls": execution_log,
        "total_results": {source: len(results) for source, results in collected_evidence.items()}
    })
  
    return state


def _execute_rag_search(endpoint_base, plan, fallback_query):
    """Execute vector database search with metadata filtering."""
    start_time = time.time()
  
    filter_params = plan.get("filters", {}).copy()
    if "category" in filter_params:
        del filter_params["category"]
  
    search_params = {
        "query": plan.get("query_text", fallback_query),
        "top_k": plan.get("top_k", DEFAULT_RESULT_LIMIT),
        "filters": filter_params
    }
  
    results = _invoke_tool(f"{endpoint_base}/{RAG_SEARCH_TOOL}", search_params)
  
    execution_metadata = {
        "tool": RAG_SEARCH_TOOL,
        "payload": search_params,
        "results_count": len(results),
        "duration_ms": int((time.time() - start_time) * 1000)
    }
  
    return results, execution_metadata


def _execute_web_search(endpoint_base, plan, fallback_query):
    """Execute live web search for real-time information."""
    start_time = time.time()
  
    search_params = {
        "query": plan.get("query_text", fallback_query),
        "top_k": min(plan.get("top_k", DEFAULT_RESULT_LIMIT), WEB_RESULT_LIMIT)
    }
  
    results = _invoke_tool(f"{endpoint_base}/{WEB_SEARCH_TOOL}", search_params)
  
    execution_metadata = {
        "tool": WEB_SEARCH_TOOL,
        "payload": search_params,
        "results_count": len(results),
        "duration_ms": int((time.time() - start_time) * 1000)
    }
  
    return results, execution_metadata


def _invoke_tool(endpoint_url, request_payload):
    """Send request to MCP tool endpoint and handle response."""
    try:
        with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
            response = client.post(endpoint_url, json=request_payload)
            response.raise_for_status()
            return response.json().get("results", [])
  
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error from {endpoint_url}: {e.response.status_code}")
        return []
    except httpx.RequestError as e:
        logger.error(f"Request failed to {endpoint_url}: {str(e)}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error during tool invocation: {str(e)}")
        return []