# E-Commerce Voice Assistant

An intelligent multi-agent voice assistant for product discovery and comparison, combining private catalog search with live web data. Built with LangGraph, ChromaDB, and modern LLM providers.

## Overview

This system provides a conversational interface for product search that:
- Processes voice or text queries
- Searches both private product catalogs (RAG) and live web sources
- Generates grounded, cited recommendations
- Validates safety and quality of responses
- Provides visual comparison tables and product cards

## Architecture

The system uses a **multi-agent workflow** built on LangGraph with five specialized agents:

1. **Intent Parser** (`intent_parser.py`) - Classifies user queries, extracts constraints (budget, material, brand), and detects safety concerns
2. **Query Strategist** (`query_strategist.py`) - Designs retrieval strategy, selects data sources, and builds filters
3. **Data Fetcher** (`data_fetcher.py`) - Executes RAG and web searches via MCP server
4. **Response Synthesizer** (`response_synthesizer.py`) - Generates grounded answers from evidence with proper citations
5. **Quality Validator** (`quality_validator.py`) - Validates safety, grounding, citations, and coherence

### Data Flow

```
User Input (Voice/Text)
    ↓
Intent Classification
    ↓
Query Planning
    ↓
Data Retrieval (RAG + Web)
    ↓
Response Synthesis
    ↓
Quality Validation
    ↓
Final Answer + UI Display
```

## Features

- **Multi-Modal Input**: Voice (Whisper ASR) or text input
- **Hybrid Search**: Combines vector search (ChromaDB) with live web search (Brave API)
- **Smart Comparison**: AI-generated comparison tables with verdicts
- **Safety Screening**: Detects and blocks unsafe queries (chemical mixing, medical advice, etc.)
- **Citation Tracking**: Proper source attribution for all claims
- **Price Reconciliation**: Identifies discrepancies between catalog and live prices
- **Visual UI**: Gradio interface with product cards, images, and comparison tables
- **Text-to-Speech**: Optional audio output for responses

## Project Structure

```
E-Commerce-Voice-Assistant/
├── graph/                      # LangGraph workflow and agents
│   ├── agents/
│   │   ├── intent_parser.py    # Intent classification
│   │   ├── query_strategist.py # Query planning
│   │   ├── data_fetcher.py     # Data retrieval
│   │   ├── response_synthesizer.py # Answer generation
│   │   └── quality_validator.py    # Quality checks
│   ├── workflow.py             # Graph construction
│   ├── state.py                # State schema
│   └── llm_interface.py        # LLM provider abstraction
├── mcp_server/                 # Model Context Protocol server
│   ├── server.py               # FastAPI endpoints
│   └── tools/
│       ├── rag_tool.py         # Vector database search
│       └── web_tool.py         # Brave Search integration
├── indexing/                   # Vector database indexing
│   └── build_index.py          # ChromaDB index builder
├── tts_asr/                    # Audio processing
│   ├── asr_whisper.py          # Speech-to-text (Whisper)
│   └── tts_client.py           # Text-to-speech (Coqui TTS)
├── UI/                         # User interface
│   └── gradio_app.py           # Gradio web app
├── prompts/                    # System prompts for agents
│   ├── system_router.md
│   ├── system_planner.md
│   ├── system_answerer.md
│   └── system_critic.md
└── requirements.txt
```

## Setup

### Installation

1. **Clone the repository** (if applicable) or navigate to the project directory

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**:
Create a `.env` file in the project root:

```env
# LLM Configuration
LLM_PROVIDER=google  # Options: google, anthropic, ollama, local
LLM_MODEL=gemini-2.5-flash-lite  # Model name
LLM_API_KEY=your_api_key_here  # Or GEMINI_API_KEY for Google
LLM_TEMPERATURE=0.3

# For Ollama/Local
# LLM_BASE_URL=http://localhost:11434/v1

# Vector Database
INDEX_PATH=./data/index
EMBED_MODEL=all-MiniLM-L6-v2
DATA_PRODUCTS=./DATA/raw/amazon_product_data_cleaned.csv

# MCP Server
MCP_BASE=http://127.0.0.1:8000

# Web Search
SEARCH_API_KEY=your_brave_api_key

# Audio Processing
ASR_MODEL=small  # Whisper model size
TTS_MODEL=tts_models/en/ljspeech/tacotron2-DDC
```

4. **Build the vector index**:
```bash
python indexing/build_index.py
```

This will:
- Load product data from CSV
- Generate embeddings using sentence-transformers
- Index documents in ChromaDB

5. **Start the MCP server** (in a separate terminal):
```bash
cd mcp_server
uvicorn server:app --host 127.0.0.1 --port 8000
```

6. **Launch the Gradio UI**:
```bash
python UI/gradio_app.py
```

The app will be available at `http://localhost:8888` (or the configured port).

## Configuration

### LLM Providers

The system supports multiple LLM providers via the `ModelInterface` class:

- **Google Gemini** (default): Set `LLM_PROVIDER=google` and `GEMINI_API_KEY`
- **Anthropic Claude**: Set `LLM_PROVIDER=anthropic` and `LLM_API_KEY`
- **Ollama**: Set `LLM_PROVIDER=ollama`, `LLM_MODEL=llama3.2`, and `LLM_BASE_URL`
- **Local Server**: Set `LLM_PROVIDER=local` and `LLM_BASE_URL`

### Vector Database

- **Embedding Model**: Change `EMBED_MODEL` in `.env` (default: `all-MiniLM-L6-v2`)
- **Index Path**: Set `INDEX_PATH` to store ChromaDB data
- **Collection Name**: Configured in `indexing/build_index.py`

### Web Search

- **Brave Search API**: Requires `SEARCH_API_KEY` from [Brave Search API](https://brave.com/search/api/)
- The system filters results to Amazon product pages only
- Scrapes real-time prices and availability

## Usage

### Web Interface

1. Open the Gradio app in your browser
2. Choose input method: **Text** or **Voice**
3. Enter your query (e.g., "Best noise cancelling headphones under $300")
4. Click **Search**
5. View results in tabs:
   - **Comparison**: AI-generated comparison table with verdict
   - **Catalog Products**: Results from private catalog
   - **Web Results**: Live web search results
   - **Citations**: Source references
   - **Execution Logs**: Agent workflow trace
   - **Audio**: Generate speech output
