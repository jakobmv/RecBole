File Descriptions:

1. movie_titles.txt:
   - Contains one movie title per line
   - Simple text file with 3,706 entries
   - Each line is a movie title from the MovieLens dataset
   - Used as a reference list of all movies in the dataset

2. embeddings.txt:
   - Tab-separated values (TSV) file
   - Two columns:
     - Column 1: Movie title (string)
     - Column 2: Embedding vector (space-separated floats)
   - Contains 3,706 rows, one for each movie
   - Each embedding is a dense vector representation of the movie
   - Format: movie_title\tfloat1 float2 float3 ... floatN
   - Used for vector similarity search and recommendation tasks

Both files are derived from the original movie_embeddings.pkl file. 