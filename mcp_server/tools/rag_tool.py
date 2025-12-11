import os
import logging
import chromadb
from chromadb.utils import embedding_functions

VECTOR_DB_PATH = os.getenv("INDEX_PATH", "./data/index")
EMBEDDING_MODEL_NAME = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
COLLECTION_NAME = "product_catalog"
TITLE_MAX_LENGTH = 220

logger = logging.getLogger(__name__)

embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=EMBEDDING_MODEL_NAME
)
db_client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
collection = db_client.get_or_create_collection(
    COLLECTION_NAME,
    embedding_function=embedding_fn
)


def _prepare_filter_clause(filters: dict | None) -> dict:
    """Transform filter dictionary into ChromaDB query format."""
    if not filters:
        return {}
  
    first_key = next(iter(filters))
    if first_key.startswith("$"):
        return filters
  
    filter_conditions = []
    for field, condition in filters.items():
        if field.startswith("$"):
            return filters
        filter_conditions.append({field: condition})
  
    if len(filter_conditions) == 1:
        return filter_conditions[0]
  
    return {"$and": filter_conditions}


def rag_search(query, top_k=5, filters=None):
    """
    Search vector database for relevant documents.
  
    Args:
        query: Search query text
        top_k: Number of results to return
        filters: Optional metadata filters
      
    Returns:
        List of matching documents with metadata
    """
    filter_clause = _prepare_filter_clause(filters)
  
    query_result = collection.query(
        query_texts=[query],
        n_results=top_k,
        where=filter_clause
    )
  
    results = []
    if not query_result["ids"] or not query_result["ids"][0]:
        return results
  
    result_count = len(query_result["ids"][0])
    for idx in range(result_count):
        metadata = query_result["metadatas"][0][idx] or {}
        document_text = query_result["documents"][0][idx]
        
        image_urls = metadata.get("image", "").split("|")[:5]  # Get the first 5 image URLs
        
        results.append({
            "doc_id": query_result["ids"][0][idx],
            "sku": metadata.get("sku"),
            "title": document_text[:TITLE_MAX_LENGTH],
            "price": metadata.get("price"),
            "rating": metadata.get("rating"),
            "brand": metadata.get("brand"),
            "ingredients": metadata.get("ingredients"),
            "image_urls": image_urls
        })
  
    return results