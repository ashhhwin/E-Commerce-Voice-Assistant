import os
import logging
import httpx

BRAVE_API_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"
REQUEST_TIMEOUT = 20
DEFAULT_RESULT_COUNT = 5

logger = logging.getLogger(__name__)


def web_search(query: str, top_k: int = DEFAULT_RESULT_COUNT):
    """
    Execute web search query and return normalized results.
  
    Args:
        query: Search query string
        top_k: Maximum number of results to return
      
    Returns:
        List of search result dictionaries
    """
    api_key = os.getenv("SEARCH_API_KEY")
  
    if not api_key:
        logger.warning("SEARCH_API_KEY environment variable not configured")
        return []

    return _execute_brave_search(query, top_k, api_key)


def _execute_brave_search(query, result_limit, api_key):
    """Execute search request via Brave Search API."""
    request_headers = {
        "Accept": "application/json",
        "X-Subscription-Token": api_key
    }

    query_params = {
        "q": query,
        "count": result_limit
    }

    try:
        with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
            response = client.get(
                BRAVE_API_ENDPOINT,
                headers=request_headers,
                params=query_params
            )
            response.raise_for_status()
            payload = response.json()
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error during search: {e.response.status_code}")
        return []
    except httpx.RequestError as e:
        logger.error(f"Request failed for query '{query}': {str(e)}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error during search: {str(e)}")
        return []

    search_results = payload.get("web", {}).get("results", [])
  
    if not search_results:
        logger.info(f"No results found for query: '{query}'")
        return []
  
    normalized_results = []
    for entry in search_results[:result_limit]:
        profile_data = entry.get("profile", {})
        extra_data = entry.get("extra_snippets", [])
      
        normalized_results.append({
            "title": entry.get("title"),
            "url": entry.get("url"),
            "snippet": entry.get("description"),
            "profile": profile_data.get("name") if profile_data else None,
            "price": entry.get("meta_url", {}).get("price"),
            "availability": extra_data[0] if extra_data else None
        })

    logger.info(f"Retrieved {len(normalized_results)} results for query: '{query}'")
    return normalized_results