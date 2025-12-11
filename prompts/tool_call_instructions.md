# Tool Call Instructions for LLM Agents

## Available Tools

### 1. `rag.search` - Private Catalog Query

**Purpose**: Search the private Amazon 2020 product catalog using semantic similarity and metadata filters.

**When to use**:
- Standard product recommendations
- Historical product data lookup
- Budget-constrained searches
- Category filtering needed

**Schema**:
```json
{
  "query": "string (required) - natural language search query",
  "top_k": "integer (optional, default: 5) - number of results",
  "filters": {
    "category": "string (optional) - e.g., 'Household Cleaning'",
    "price": {
      "$lte": "float (optional) - maximum price",
      "$gte": "float (optional) - minimum price"
    }
  }
}
```

**Returns**:
```json
{
  "tool": "rag.search",
  "timestamp": 1234567890.0,
  "results": [
    {
      "doc_id": "A001",
      "sku": "12345",
      "title": "Product name and brief description",
      "price": 12.49,
      "rating": 0.0,
      "brand": "",
      "ingredients": "",
      "category": "Household Cleaning"
    }
  ]
}
```

**Example call**:
```json
{
  "query": "eco-friendly stainless steel cleaner",
  "top_k": 5,
  "filters": {
    "category": "Household Cleaning",
    "price": {"$lte": 15.0}
  }
}
```

**Limitations**:
- Rating always 0.0 (not in dataset)
- Brand often empty
- Ingredients often empty
- Price may be historical

---

### 2. `web.search` - Live Web Query

**Purpose**: Search current web results for real-time pricing, availability, and product information.

**When to use**:
- User asks for "current", "today", "latest", "now"
- Availability/stock status needed
- Price verification against market
- Comparison with live data

**Schema**:
```json
{
  "query": "string (required) - search query",
  "top_k": "integer (optional, default: 5) - number of results"
}
```

**Returns**:
```json
{
  "tool": "web.search",
  "timestamp": 1234567890.0,
  "results": [
    {
      "title": "Product title from web",
      "url": "https://example.com/product",
      "snippet": "Description or excerpt",
      "profile": "website name",
      "price": null,
      "availability": null
    }
  ]
}
```

**Example call**:
```json
{
  "query": "Lysol disinfectant spray current price",
  "top_k": 3
}
```

**Rate limits**:
- Limit to 3-5 results per query
- Results cached for 60-300 seconds
- Respect API rate limits

**Limitations**:
- price/availability often null (structured data not always available)
- Requires parsing from snippet
- URL might not be direct product page

---

## Tool Call Best Practices

### 1. Always Log
Every tool call must be logged:
```json
{
  "node": "<agent_name>",
  "tool": "<tool_name>",
  "timestamp": "<ISO datetime>",
  "payload": {<request payload>},
  "response_summary": "<number of results, any errors>"
}
```

**NEVER log**:
- API keys
- Authentication tokens
- User PII

### 2. Error Handling

**If tool returns empty results**:
- Do NOT hallucinate data
- Acknowledge: "No results found"
- Suggest alternatives or broader search

**If tool call fails**:
- Log the error
- Retry once (with backoff)
- If still fails, gracefully degrade
- Inform user: "I'm having trouble accessing live data"

### 3. Reconciliation Logic

When both tools are called:

**Step 1: Match products**
- Use fuzzy string matching on titles (>80% similarity)
- Consider brand/SKU if available

**Step 2: Compare prices**
- If prices differ by >10%: mention both
- If prices similar (±5%): use one
- Prefer RAG for historical, web for current

**Step 3: Flag conflicts**
- Log discrepancies
- Present both to user
- Don't pick one arbitrarily

**Example**:
```
RAG: "Product X - $12.99"
Web: "Product X - $14.49"
Difference: 11.5%

Answer: "Our catalog shows twelve ninety-nine, but it's currently fourteen forty-nine online. Price may have increased."
```

### 4. Result Ranking

**After retrieval, rank by**:
1. Relevance score (from vector search)
2. Price (if budget query)
3. Rating (if available)
4. Price-per-oz (if value query)

**Always show top 3 max** in voice response
**Display full results** on screen

### 5. Citation Tracking

Track source for every fact:
```json
{
  "claim": "Product X costs $12.99",
  "source": "rag",
  "doc_id": "A001",
  "confidence": "high"
}
```

```json
{
  "claim": "Currently in stock at Walmart",
  "source": "web",
  "url": "https://walmart.com/product",
  "confidence": "medium"
}
```

### 6. Caching Strategy

**RAG results**: No caching (data is static)
**Web results**: Cache for 60-300 seconds
- Key: hash of query string
- Reduces API calls
- Balances freshness vs cost

## Debugging Tool Calls

**If results seem wrong**:
1. Check query text (too broad/narrow?)
2. Verify filters are correct format
3. Check top_k value (too low?)
4. Review embedding model alignment

**If no matches**:
1. Try broader query
2. Remove filters
3. Check category spelling
4. Verify data exists in index

**If web search fails**:
1. Check API key configured
2. Verify internet connection
3. Check rate limits not exceeded
4. Try simpler query

## Schema Validation

Before calling tools, validate:
- ✅ All required fields present
- ✅ Types are correct (string, int, float)
- ✅ Filters use valid operators ($lte, $gte)
- ✅ top_k is reasonable (1-10)

Invalid calls waste tokens and time.
