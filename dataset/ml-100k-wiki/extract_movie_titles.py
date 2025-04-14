# This script extracts all movie titles from the movie_embeddings.pkl file and saves them to a new file called movie_titles.txt
# Note that you must use the numpy version for the pickle process to work..

import pickle
import os
import numpy as np

# Load the pickle file
with open("dataset/ml-100k-wiki/movie_embeddings.pkl", "rb") as f:
    embeddings_dict = pickle.load(f)

# Extract all movie titles (keys)
movie_titles = list(embeddings_dict.keys())

# Save movie titles to a new file
with open("dataset/ml-100k-wiki/movie_titles.txt", "w", encoding="utf-8") as f:
    for title in movie_titles:
        f.write(f"{title}\n")

print(f"Extracted {len(movie_titles)} movie titles to movie_titles.txt")
