"""
This script creates links.tsv by matching movie titles between:
1. ml-100k-wiki.itemOld (original dataset, copy of ml-100k.item)
2. movie_titles.txt (from embeddings)

The output file (links.tsv) contains:
item_id    movie_title_item    movie_title_emb
where movie_title_emb is the matching title from embeddings.txt
"""

import os

# Get the directory path of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Read item file
item_data = {}
with open(os.path.join(script_dir, "ml-100k-wiki.itemOld"), "r", encoding="utf-8") as f:
    # Skip header
    next(f)
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) >= 2:
            item_id = parts[0]
            movie_title = parts[1]
            item_data[item_id] = movie_title

print(f"Loaded {len(item_data)} items from ml-100k-wiki.itemOld")

# Read embeddings titles
embedding_titles = set()
with open(os.path.join(script_dir, "movie_titles.txt"), "r", encoding="utf-8") as f:
    for line in f:
        title = line.strip()
        embedding_titles.add(title)

print(f"Loaded {len(embedding_titles)} titles from movie_titles.txt")

# Create links.tsv
with open(os.path.join(script_dir, "links.tsv"), "w", encoding="utf-8") as f:
    f.write("item_id\tmovie_title_item\tmovie_title_emb\n")  # Header

    matches = 0
    for item_id, item_title in item_data.items():
        # Check if the exact title exists in embeddings
        if item_title in embedding_titles:
            f.write(f"{item_id}\t{item_title}\t{item_title}\n")
            matches += 1
            continue

        # If no exact match, look for close matches
        matched = False
        for emb_title in embedding_titles:
            # Simple matching - check if the item title is in the embedding title
            # or vice versa (you may want to improve this matching logic)
            if item_title in emb_title or emb_title in item_title:
                f.write(f"{item_id}\t{item_title}\t{emb_title}\n")
                matched = True
                matches += 1
                break

        # If no match found, still record it but with empty embedding title
        if not matched:
            f.write(f"{item_id}\t{item_title}\t\n")

print(f"Created links.tsv with {matches} matches out of {len(item_data)} items")
