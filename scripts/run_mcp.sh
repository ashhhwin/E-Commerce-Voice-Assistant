#!/bin/bash

# Load environment variables from .env into the shell
export $(grep -v '^#' .env | xargs)

# Start the MCP server on port 8000 with auto-reload
uvicorn mcp_server.server:app --reload --port 8000
