import os
import logging
import httpx
from bs4 import BeautifulSoup
import time

BRAVE_API_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"
REQUEST_TIMEOUT = 30
DEFAULT_RESULT_COUNT = 10

# Fake User-Agent to help bypass basic blocking by Amazon
HEADERS_SCRAPER = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}

logger = logging.getLogger(__name__)

def Web_Search(query: str, top_k: int = 5):
    """
    Execute web search query specifically for Amazon and return normalized results.
    """
    api_key = os.getenv("SEARCH_API_KEY", "BSA_xrFcUVUty7k6l8zN_kYSJchJ2G_")
    
    if not api_key:
        logger.warning("SEARCH_API_KEY environment variable not configured")
        return []

    # Modify query to force Amazon results
    targeted_query = f"site:amazon.com {query}"
    
    # Request slightly more than needed to account for potential filtering
    raw_results = _execute_brave_search(targeted_query, top_k + 5, api_key)
    
    # Filter to ensure URL actually contains amazon.com
    amazon_results = [r for r in raw_results if "amazon.com" in r.get('url', '')]
    
    # Trim to user limit
    amazon_results = amazon_results[:top_k]

    normalized_results = []

    for entry in amazon_results:
        # Default values
        price = "N/A"
        availability = "Check Site"
        profile_name = entry.get("profile", {}).get("name") if entry.get("profile") else "Amazon"
        
        url = entry.get("url")
        
        # Attempt to visit the page to get Price/Availability
        try:
            # Random sleep to minimize blocking risk
            time.sleep(1.5) 
            
            with httpx.Client(timeout=10, headers=HEADERS_SCRAPER, follow_redirects=True) as client:
                response = client.get(url)
                
                # Only parse if we got a valid page (Amazon often returns 503 to bots)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, "html.parser")

                    # 1. Attempt to find Price
                    # Amazon uses multiple classes for price; checking the most common ones
                    price_element = soup.select_one('.a-price .a-offscreen')
                    if not price_element:
                        price_element = soup.select_one('.a-price-whole')
                    
                    if price_element:
                        price = price_element.get_text(strip=True)

                    # 2. Attempt to find Availability
                    # Look for the #availability div or common text
                    avail_text = soup.get_text().lower()
                    if "in stock" in avail_text:
                        availability = "In Stock"
                    elif "currently unavailable" in avail_text:
                        availability = "Out of Stock"
                    elif "only" in avail_text and "left in stock" in avail_text:
                         availability = "Low Stock"

        except Exception as e:
            logger.error(f"Error scraping {url}: {str(e)}")
            # We continue even if scraping fails, returning the API data we have
        
        normalized_results.append({
            "title": entry.get("title"),
            "url": url,
            "snippet": entry.get("description"),
            "profile": profile_name,
            "price": price,
            "availability": availability
        })

    return normalized_results

def _execute_brave_search(query, count, api_key):
    """Execute search request via Brave Search API."""
    headers = {
        "Accept": "application/json",
        "X-Subscription-Token": api_key
    }
    
    try:
        with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
            response = client.get(
                BRAVE_API_ENDPOINT,
                headers=headers,
                params={"q": query, "count": count}
            )
            response.raise_for_status()
            return response.json().get("web", {}).get("results", [])
    except Exception as e:
        logger.error(f"Search API failed: {e}")
        return []

def main():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    query = input("Enter product name: ")
    if not query:
        query = "laptop"
        
    # User requested at least 3-4 results
    results = Web_Search(query, top_k=5)
    
    if results:
        print(f"\nFound {len(results)} Amazon results:\n")
        for idx, result in enumerate(results, 1):
            print(f"Result {idx}:")
            print(f"Title: {result['title']}")
            print(f"URL: {result['url']}")
            print(f"Snippet: {result['snippet']}")
            print(f"Profile: {result['profile']}")
            print(f"Price: {result['price']}")
            print(f"Availability: {result['availability']}")
            print("-" * 50)
    else:
        print("No results found.")

if __name__ == "__main__":
    main()