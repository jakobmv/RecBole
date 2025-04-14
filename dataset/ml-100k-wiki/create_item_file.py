"""
This script creates the ml-100k-wiki.item file for RecBole by:
1. Reading links.tsv to get mappings between item_ids and movie titles
2. Reading embeddings.txt to get Wikipedia embeddings for each movie
3. Creating a new item file with format:
   item_id:token    movie_title:token_seq    wiki_embedding:float_seq
"""

import os

# Get the directory path of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Read links file to get item_id to movie_title mapping
links_data = {}
with open(os.path.join(script_dir, "links.tsv"), "r", encoding="utf-8") as f:
    next(f)  # Skip header
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) >= 3:  # item_id, movie_title_item, movie_title_emb
            item_id = parts[0]
            movie_title = parts[1]
            emb_title = parts[2]
            if emb_title:  # Only store if we have a matching embedding title
                links_data[item_id] = (movie_title, emb_title)

print(f"Loaded {len(links_data)} items from links.tsv")

# Read embeddings file to get title to embedding mapping
embeddings_data = {}
with open(os.path.join(script_dir, "embeddings.txt"), "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) == 2:  # title, embedding
            title = parts[0]
            embedding = parts[1]  # Space-separated float values
            embeddings_data[title] = embedding

print(f"Loaded {len(embeddings_data)} embeddings from embeddings.txt")

# Create the new item file
with open(os.path.join(script_dir, "ml-100k-wiki.item"), "w", encoding="utf-8") as f:
    # Write header
    f.write("item_id:token\tmovie_title:token_seq\twiki_embedding:float_seq\n")

    # Write data
    matches = 0
    for item_id, (movie_title, emb_title) in links_data.items():
        if emb_title in embeddings_data:
            embedding = embeddings_data[emb_title]
            f.write(f"{item_id}\t{movie_title}\t{embedding}\n")
            matches += 1

print(f"Created ml-100k-wiki.item with {matches} items")
