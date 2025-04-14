"""
This script reads embeddings.txt and prints the dimension of the embeddings.
It reads the first line and counts the number of space-separated float values.
"""

import os

# Get the directory path of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Read first line of embeddings file
with open(os.path.join(script_dir, "embeddings.txt"), "r", encoding="utf-8") as f:
    first_line = f.readline().strip()
    parts = first_line.split("\t")
    if len(parts) == 2:
        embedding = parts[1]  # Get the embedding part
        dim = len(embedding.split())  # Count space-separated values
        print(f"Embedding dimension: {dim}")
    else:
        print("Error: Unexpected file format")
