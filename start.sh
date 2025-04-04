#!/bin/bash

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Ensure dependencies are installed
pip install --no-cache-dir -r requirements.txt

# Start FastAPI app
exec uvicorn app.main:app --host 0.0.0.0 --port 10000

