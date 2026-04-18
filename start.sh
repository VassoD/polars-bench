#!/bin/bash
set -e
python data/make_data.py
uvicorn server:app --host 0.0.0.0 --port 8000
