#!/bin/sh

echo "📦 Downloading models from Google Drive..."
python models/download_models.py

# Start FastAPI app
echo "🚀 Starting FastAPI..."
exec uvicorn app.main:app --host 0.0.0.0 --port 8000

