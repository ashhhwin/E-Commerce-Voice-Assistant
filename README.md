# Agentic Voice-to-Voice Product Discovery Assistant

**Final Project - Generative AI Course**

End-to-end voice-to-voice AI assistant for e-commerce product discovery using multi-agent orchestration, RAG, and MCP tools.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2.37-green.svg)](https://langchain-ai.github.io/langgraph/)

---

## 🎯 Project Overview

This system implements a sophisticated **multi-agent AI assistant** that:

- 🎤 **Accepts voice queries** via Whisper ASR
- 🧠 **Reasons with 5 specialized agents** orchestrated by LangGraph  
- 📚 **Retrieves from private catalog** (Amazon 2020 dataset via RAG)
- 🌐 **Augments with live web data** when needed (Brave Search)
- ✅ **Ensures grounding & safety** with citation tracking and validation
- 🔊 **Responds naturally** via OpenAI TTS

**Example interaction**:
```
User (voice): "Recommend an eco-friendly stainless steel cleaner under $15"
           ↓
    [Agent Pipeline processes request]
           ↓
System (voice): "I found two eco-friendly options. My top pick is EcoShine 
                 Steel Polish at $12.49—plant-based formula. See details 
                 on screen. (Sources: doc #A001, doc #A002)"
```

---

## 🚀 Quick Start (uv venv)

```bash
# 1) Install uv (if missing)
pip install uv  # or brew install uv

# 2) Create & activate venv (mac/Linux)
uv venv .venv
source .venv/bin/activate

# 3) Install deps with uv
uv pip install -r requirements.txt
brew install ffmpeg  # required for Whisper

# 4) Configure
cp configs/env.example .env
# Edit .env: set OPENAI_API_KEY
# Optional for Kaggle CLI: set KAGGLE_USERNAME and KAGGLE_KEY

# 5) Build index (sample data or Kaggle data)
bash scripts/build_index.sh

# 6) Run (2 terminals)
# Terminal 1
PYTHONPATH="$PWD" .venv/bin/uvicorn mcp_server.server:app --host 127.0.0.1 --port 8000
# Terminal 2
PYTHONPATH="$PWD" .venv/bin/streamlit run app/ui_streamlit.py --server.port 8501

# 7) Open browser: http://localhost:8501
```

---

## 📊 System Architecture

### Multi-Agent Pipeline (LangGraph)

```
┌────────────┐    ┌───────────┐    ┌──────────────┐
│   ROUTER   │───▶│  PLANNER  │───▶│  RETRIEVER   │
│            │    │           │    │              │
│ Extract    │    │ Design    │    │ Call MCP     │
│ Intent     │    │ Strategy  │    │ Tools        │
└────────────┘    └───────────┘    └──────────────┘
                                            │
                                            ▼
                                    ┌──────────────┐
                                    │   MCP TOOLS  │
                                    ├──────────────┤
                                    │ rag.search   │
                                    │ web.search   │
                                    └──────────────┘
                                            │
        ┌───────────┐    ┌───────────┐    ▼
        │  CRITIC   │◀───│ ANSWERER  │◀───┘
        │           │    │           │
        │ Validate  │    │ Synthesize│
        │ Ground    │    │ Response  │
        └───────────┘    └───────────┘
```

### Agent Responsibilities

| Agent | LLM Used | Purpose | Key Features |
|-------|----------|---------|-------------|
| **Router** | ✓ | Extract intent, constraints, safety flags | JSON output, regex fallback |
| **Planner** | ✓ | Decide sources, build filters, set ranking | Strategy design, rule fallback |
| **Retriever** | ✗ | Execute MCP tool calls | HTTP error handling, logging |
| **Answerer** | ✓ | Synthesize grounded response | Reconciliation, citations |
| **Critic** | ✗ | Validate safety, grounding, citations | 6-check validation system |

**See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed diagrams and component specifications.**

---

## 📁 Project Structure

```
agentic-voice-assistant/
├── app/                             # Streamlit UI
│   ├── ui_streamlit.py              # Main application
│   ├── audio_utils.py               # Audio processing
│   └── components.py                # UI components
├── configs/
│   └── env.example                  # Configuration template
├── data/
│   ├── DATASET_SETUP.md             # Dataset download guide
│   ├── processed/                   # Processed CSV
│   └── index/                       # ChromaDB vector store
├── graph/                           # Agent pipeline
│   ├── langgraph_pipeline.py        # LangGraph orchestration
│   ├── llm_client.py                # Model-agnostic LLM interface
│   ├── schemas.py                   # State schemas
│   └── nodes/                       # 5 agent implementations
│       ├── router.py
│       ├── planner.py
│       ├── retriever.py
│       ├── answerer.py
│       └── critic.py
├── indexing/
│   └── build_index.py               # Vector index creation
├── mcp_server/                      # MCP tool server
│   ├── server.py                    # FastAPI server
│   └── tools/
│       ├── rag_tool.py              # Private catalog search
│       └── web_tool.py              # Web search
├── prompts/                         # ⭐ Full prompt disclosure
│   ├── system_router.md             # 80+ lines
│   ├── system_planner.md            # 100+ lines
│   ├── system_answerer.md           # 120+ lines
│   ├── system_critic.md             # 90+ lines
│   ├── tool_call_instructions.md    # 150+ lines
│   └── few_shots.jsonl              # 5 examples
├── scripts/
│   ├── build_index.sh               # Build vector index
│   ├── run_mcp.sh                   # Start MCP server
│   └── run_ui.sh                    # Start UI
├── tts_asr/
│   ├── asr_whisper.py               # Whisper ASR
│   └── tts_client.py                # OpenAI TTS
├── ARCHITECTURE.md                  # Detailed system design
├── DEMO_GUIDE.md                    # Setup & demo instructions
├── SAFETY.md                        # Safety considerations
└── requirements.txt                 # Dependencies
```

---

## 🔧 Installation & Setup

### Prerequisites

- Python 3.10+
- ffmpeg (for Whisper ASR): `brew install ffmpeg`
- 4GB+ RAM
- OpenAI API key

### Step-by-Step (uv-first)

```bash
# 1. Clone and create environment
git clone <repo-url>
cd agentic-voice-assistant
pip install uv  # if uv not installed
uv venv .venv
source .venv/bin/activate

# 2. Install dependencies
uv pip install -r requirements.txt

# 3. Install ffmpeg
brew install ffmpeg  # macOS
# OR: sudo apt-get install ffmpeg  # Linux

# 4. Configure API keys
cp configs/env.example .env
nano .env  # Required: OPENAI_API_KEY
           # Optional: SEARCH_API_KEY (Brave for web search), KAGGLE_USERNAME, KAGGLE_KEY

# 5. Build vector index
# Option A: Quick start (3 sample items)
PYTHONPATH="$PWD" .venv/bin/python indexing/build_index.py

# Option B: Full dataset (10,002 items from Kaggle)
# First: Download from Kaggle (see data/DATASET_SETUP.md)
# Then:
DATA_PRODUCTS="./data/raw/home/sdf/marketing_sample_for_amazon_com-ecommerce__20200101_20200131__10k_data.csv" \
  PYTHONPATH="$PWD" .venv/bin/python indexing/build_index.py

# 6. Verify setup
.venv/bin/python - <<'PY'
import chromadb
client = chromadb.PersistentClient(path='./data/index')
col = client.get_collection('amazon2020')
print(f'✓ Indexed {col.count()} documents')
PY
```

---

## 🎮 Usage

### Start Services

**Terminal 1: MCP Server**
```bash
bash scripts/run_mcp.sh
# Expected: INFO: Uvicorn running on http://127.0.0.1:8000
```

**Terminal 2: Streamlit UI**
```bash
bash scripts/run_ui.sh
# Expected: Local URL: http://localhost:8501
```

### Using the Interface

1. Open browser: http://localhost:8501
2. Choose input: Voice recording OR typed text
3. Submit query: Click "Transcribe & Search"
4. View results:
   - Agent step log (decision transparency)
   - Product table with details
   - Citations (doc IDs + web URLs)
5. Play TTS: Click "🔊 Play TTS"

### Example Queries

```
✅ "Recommend an eco-friendly stainless steel cleaner under $15"
   → RAG only, budget filter, sorted by price

✅ "What's the current price of Lysol spray in stock?"
   → RAG + Web, price comparison, availability check

✅ "Find Scotch-Brite heavy duty scrub pads"
   → Brand-specific search, semantic matching

❌ "Can I mix bleach and ammonia?"
   → Safety rejection, refusal message
```

---

## 📝 Prompts & Agent Design

### Prompt Disclosure (Grading Requirement)

All prompts in `prompts/` directory:

| File | Purpose | Lines |
|------|---------|-------|
| `system_router.md` | Intent extraction, safety screening | 80+ |
| `system_planner.md` | Source selection, filter strategy | 100+ |
| `system_answerer.md` | Response synthesis, grounding rules | 120+ |
| `system_critic.md` | Quality validation, safety checks | 90+ |
| `tool_call_instructions.md` | MCP tool schemas & best practices | 150+ |
| `few_shots.jsonl` | Complete query examples | 5 |

### Design Principles

1. **Grounding**: Every claim traces to evidence
2. **Citations**: doc_id for private, URL for web
3. **Safety**: Deny chemical mixing, medical advice
4. **Transparency**: Log all decisions
5. **Fallbacks**: Regex/templates if LLM fails

---

## 🎬 Demo & Evaluation

### 7-Minute Demo Script

See `DEMO_GUIDE.md` for complete guide.

**Outline**:
1. Introduction (1 min)
2. Architecture (1 min)
3. Simple query (1.5 min)
4. Hybrid query (1.5 min)
5. Safety demo (0.5 min)
6. Prompt disclosure (0.5 min)
7. Results & limitations (1 min)

### Testing

```bash
# Test MCP tools
curl -X POST http://127.0.0.1:8000/rag.search \
  -H "Content-Type: application/json" \
  -d '{"query":"cleaner","top_k":3}'

# Test components
python -c "from tts_asr.asr_whisper import transcribe; print('✓ Whisper')"
python -c "from graph.llm_client import get_llm_client; llm = get_llm_client(); print('✓ LLM')"
```

---

## ⚙️ Configuration

### LLM Providers (Model-Agnostic)

**OpenAI** (default):
```bash
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=sk-...
```

**Anthropic**:
```bash
LLM_PROVIDER=anthropic
LLM_MODEL=claude-3-5-sonnet-20241022
ANTHROPIC_API_KEY=sk-ant-...
```

**Local models** (vLLM, Ollama):
```bash
LLM_PROVIDER=local
LLM_MODEL=llama-3.1-8b
LLM_BASE_URL=http://localhost:11434/v1
```

### Web Search (Optional)

The system can augment RAG results with live web search when queries indicate a need for current information (e.g., "today", "current", "latest").

**Enable Brave Search** (recommended):
1. Get API key: https://api.search.brave.com
2. Add to `.env`:
   ```bash
   SEARCH_API_KEY=<your-brave-api-key>
   SEARCH_PROVIDER=brave
   ```
3. Test:
   ```bash
   curl -X POST http://127.0.0.1:8000/web.search \
     -H "Content-Type: application/json" \
     -d '{"query":"best product today","top_k":5}'
   ```

If `SEARCH_API_KEY` is not set, web search is gracefully disabled (queries fall back to RAG only).

---

## 📊 Results

### What Works ✅

- End-to-end voice workflow
- Multi-agent reasoning (LangGraph)
- Grounded answers with citations
- Model-agnostic (OpenAI, Claude tested)
- MCP server (2 tools: rag.search, web.search)
- Live web search with Brave API (optional)
- Safety checks (chemical mixing, medical)
- Transparent agent logs

### Limitations ⚠️

- **RAG dataset**: 10,002 items (sample from Kaggle); doesn't cover all product categories
  - Missing items: Web search fallback fills gaps (e.g., "rice cooker" → returns Brave results)
- Dataset lacks ratings/reviews
- Fragment TTS (not streaming)
- Sequential agents (no parallelism)
- Basic title matching
- Stateless queries

### Future Work 🚧

- Streaming audio (OpenAI Realtime)
- Multi-turn conversations
- Advanced RAG (reranking, expansion)
- Product images, comparison tables
- User preference memory
- Production deployment

---

## 📚 Documentation

- **[ARCHITECTURE.md](ARCHITECTURE.md)**: System design, data flow, components
- **[DEMO_GUIDE.md](DEMO_GUIDE.md)**: Setup, demo script, troubleshooting
- **[data/DATASET_SETUP.md](data/DATASET_SETUP.md)**: Dataset download guide
- **[SAFETY.md](SAFETY.md)**: Safety considerations
- **[prompts/](prompts/)**: Complete prompt disclosure

---

## 📜 License

MIT License

---

## 🙏 Acknowledgments

- Amazon Product Dataset 2020 (Kaggle)
- LangGraph (agent orchestration)
- ChromaDB (vector database)
- OpenAI (Whisper, GPT, TTS)
- Brave (web search API)

---

**Project**: Agentic Voice-to-Voice Product Discovery  
**Course**: Generative AI, University of Chicago  
**Date**: December 2025
