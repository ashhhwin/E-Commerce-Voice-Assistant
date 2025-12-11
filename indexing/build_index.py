import os
import re
import logging
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions

PRODUCT_DATA_PATH = os.getenv("DATA_PRODUCTS", "./data/raw/amazon_product_data_cleaned.csv")
VECTOR_DB_PATH = os.getenv("INDEX_PATH", "./data/index")
EMBEDDING_MODEL_NAME = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
COLLECTION_NAME = "product_catalog"
BATCH_SIZE = 1000

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def _extract_price_per_ounce(price, product_features):
    """Calculate price per ounce from product features text."""
    try:
        ounce_match = re.search(r"([\d.]+)\s*oz", str(product_features), re.IGNORECASE)
        if not ounce_match:
            return None
        
        ounce_value = float(ounce_match.group(1))
        if price and ounce_value:
            return float(price) / ounce_value
        return None
    except (ValueError, TypeError, AttributeError):
        return None


def _sanitize_metadata(value):
    """Convert metadata values to ChromaDB-compatible types."""
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    if isinstance(value, (list, dict, tuple)):
        return str(value)
    return value


def _transform_dataframe_to_documents(dataframe: pd.DataFrame):
    """Transform DataFrame rows into document tuples for indexing."""
    column_mapping = {
        "Uniq Id": "id",
        "Product Name": "title",
        "Category": "category",
        "Selling Price": "price",
        "About Product": "features",
        "Image": "image"
    }
    
    dataframe = dataframe.rename(columns=column_mapping)
    
    dataframe["price"] = dataframe["price"].astype(str).str.replace("$", "", regex=False)
    dataframe["price"] = pd.to_numeric(dataframe["price"], errors="coerce")
    
    dataframe["price_per_oz"] = dataframe.apply(
        lambda row: _extract_price_per_ounce(row.get("price"), row.get("features")),
        axis=1
    )
    
    documents = []
    for _, row in dataframe.iterrows():
        document_text = " ".join([
            str(row.get("title", "")),
            str(row.get("features", "")),
            str(row.get("category", "")),
        ])
        
        price_value = row.get("price")
        price_per_oz_value = row.get("price_per_oz")
        
        metadata = {
            "sku": _sanitize_metadata(row.get("id")),
            "title": _sanitize_metadata(row.get("title", "")),
            "brand": "",
            "category": _sanitize_metadata(row.get("category")),
            "price": float(price_value) if not pd.isna(price_value) else 0.0,
            "rating": 0.0,
            "ingredients": "",
            "price_per_oz": (
                float(price_per_oz_value)
                if price_per_oz_value and not pd.isna(price_per_oz_value)
                else 0.0
            ),
            "image": _sanitize_metadata(row.get("image"))
        }
        
        metadata = {key: _sanitize_metadata(val) for key, val in metadata.items()}
        
        documents.append((str(row.get("id")), document_text, metadata))
    
    return documents


def _batch_iterable(items, batch_size):
    """Split iterable into fixed-size batches."""
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def main():
    try:
        logger.info(f"Loading product data from {PRODUCT_DATA_PATH}")
        dataframe = pd.read_csv(PRODUCT_DATA_PATH, encoding='latin1')  # or use chardet or errors parameter
        logger.info(f"Loaded {len(dataframe)} products")
    except FileNotFoundError:
        logger.error(f"Product data file not found: {PRODUCT_DATA_PATH}")
        return
    except Exception as e:
        logger.error(f"Failed to load product data: {str(e)}")
        return
    
    try:
        os.makedirs(VECTOR_DB_PATH, exist_ok=True)
        logger.info(f"Initializing vector database at {VECTOR_DB_PATH}")
        
        db_client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
        
        embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL_NAME
        )
        logger.info(f"Using embedding model: {EMBEDDING_MODEL_NAME}")
        
        collection = db_client.get_or_create_collection(
            COLLECTION_NAME,
            embedding_function=embedding_fn
        )
    except Exception as e:
        logger.error(f"Failed to initialize database or embedding model: {str(e)}")
        return
    
    try:
        logger.info("Transforming products into documents")
        documents = _transform_dataframe_to_documents(dataframe)
        document_ids, document_texts, document_metadata = zip(*documents)
        logger.info(f"Prepared {len(documents)} documents for indexing")
    except Exception as e:
        logger.error(f"Failed to transform documents: {str(e)}")
        return
    
    try:
        existing_data = collection.get()
        if len(existing_data["ids"]) > 0:
            logger.info(f"Clearing {len(existing_data['ids'])} existing documents")
            collection.delete(ids=existing_data["ids"])
    except Exception as e:
        logger.error(f"Failed to clear existing collection: {str(e)}")
        return
    
    try:
        total_batches = (len(document_ids) + BATCH_SIZE - 1) // BATCH_SIZE
        logger.info(f"Indexing documents in {total_batches} batches of {BATCH_SIZE}")
        
        batch_number = 0
        for id_batch, text_batch, meta_batch in zip(
            _batch_iterable(list(document_ids), BATCH_SIZE),
            _batch_iterable(list(document_texts), BATCH_SIZE),
            _batch_iterable(list(document_metadata), BATCH_SIZE)
        ):
            batch_number += 1
            collection.add(
                ids=list(id_batch),
                documents=list(text_batch),
                metadatas=list(meta_batch)
            )
            logger.info(f"Indexed batch {batch_number}/{total_batches} ({len(id_batch)} documents)")
        
        logger.info(f"Successfully indexed {len(document_ids)} documents into collection '{COLLECTION_NAME}'")
    except Exception as e:
        logger.error(f"Failed during indexing: {str(e)}")
        return


if __name__ == "__main__":
    main()