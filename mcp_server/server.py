import os, time
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from mcp_server.tools.rag_tool import rag_search
from mcp_server.tools.web_tool import web_search

load_dotenv()

app = FastAPI(title="Product MCP Server")

class RagQuery(BaseModel):
    query: str
    top_k: int = 5
    filters: dict | None = None

class WebQuery(BaseModel):
    query: str
    top_k: int = 5

@app.post("/rag.search")
def rag_endpoint(q: RagQuery):
    results = rag_search(q.query, q.top_k, q.filters)
    return {"tool":"rag.search","timestamp":time.time(),"results":results}

@app.post("/web.search")
def web_endpoint(q: WebQuery):
    results = web_search(q.query, q.top_k)
    return {"tool":"web.search","timestamp":time.time(),"results":results}
