#!/bin/bash
echo "Training model..."
python model.py
echo "Starting server..."
uvicorn main:app --host 0.0.0.0 --port $PORT