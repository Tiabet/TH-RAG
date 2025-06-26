#!/bin/bash

echo "Step 1: Construct graph JSON from text chunks..."
python graph_construction.py

echo "Step 2: Convert JSON to GEXF..."
python json_to_gexf_v2.py

echo "Step 3: Build edge embeddings and FAISS index..."
python edge_embedding.py