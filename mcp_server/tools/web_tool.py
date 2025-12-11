import os
import logging
import httpx
import time
import re  # Added for stricter URL pattern matching
from bs4 import BeautifulSoup

BRAVE_API_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"
REQUEST_TIMEOUT = 20
DEFAULT_RESULT_COUNT = 5

# Fake User-Agent to help bypass basic blocking by Amazon
HEADERS_SCRAPER = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}

logger = logging.getLogger(__name__)

def web_search(query: str, top_k: int = DEFAULT_RESULT_COUNT):
    """
    Execute web search query specifically for Amazon and return normalized results.
    Strictly filters for specific product pages (removing search/category pages).
    """
    api_key = os.getenv("SEARCH_API_KEY")
  
    if not api_key:
        logger.warning("SEARCH_API_KEY environment variable not configured")
        return []

    # 1. Modify query to force Amazon results
    targeted_query = f"site:amazon.com {query}"

    # 2. Get Raw Results 
    # We fetch a larger buffer (top_k * 3) because we expect to discard many
    # generic "search result" or "category" pages during the filtering step.
    raw_results = _execute_brave_search(targeted_query, top_k * 3, api_key)

    # 3. Filter for valid Amazon PRODUCT URLs only
    # Valid product URLs typically contain '/dp/' or '/gp/product/'
    # We explicitly exclude '/s?' (search) and '/b?' (browse) pages
    amazon_product_results = []
    
    for r in raw_results:
        url = r.get('url', '')
        if "amazon.com" in url:
            # Check for specific product markers
            if ("/dp/" in url or "/gp/product/" in url) and not ("/s?" in url or "/b?" in url):
                amazon_product_results.append(r)
    
    # Trim to the requested limit
    amazon_results = amazon_product_results[:top_k]
    
    normalized_results = []
    
    # 4. Scrape details for each result
    for entry in amazon_results:
        url = entry.get("url")
        price = "N/A"
        availability = "Check Site"
        profile_name = entry.get("profile", {}).get("name") if entry.get("profile") else "Amazon"
        
        # Attempt to scrape real-time data
        try:
            time.sleep(1.0) # Polite delay
            with httpx.Client(timeout=10, headers=HEADERS_SCRAPER, follow_redirects=True) as client:
                response = client.get(url)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, "html.parser")
                    
                    # Try finding price
                    price_element = soup.select_one('.a-price .a-offscreen')
                    if not price_element:
                        price_element = soup.select_one('.a-price-whole')
                    if price_element:
                        price = price_element.get_text(strip=True)
                        
                    # Try finding availability
                    text_content = soup.get_text().lower()
                    if "in stock" in text_content:
                        availability = "In Stock"
                    elif "currently unavailable" in text_content:
                        availability = "Out of Stock"
                    elif "left in stock" in text_content:
                        availability = "Low Stock"
        except Exception as e:
            logger.error(f"Scraping failed for {url}: {e}")
            # If scraping fails, fall back to API metadata
            meta_price = entry.get("meta_url", {}).get("price")
            if meta_price: 
                price = meta_price

        normalized_results.append({
            "title": entry.get("title"),
            "url": url,
            "snippet": entry.get("description"),
            "profile": profile_name,
            "price": price,
            "availability": availability
        })

    logger.info(f"Retrieved {len(normalized_results)} Amazon product results for query: '{query}'")
    return normalized_results


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

    return payload.get("web", {}).get("results", [])