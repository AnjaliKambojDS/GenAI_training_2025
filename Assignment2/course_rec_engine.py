import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss  

# Load dataset
data_url = "https://raw.githubusercontent.com/Bluedata-Consulting/GAAPB01-training-codebase/refs/heads/main/Assignments/assignment2dataset.csv"
courses_df = pd.read_csv(data_url)

# Initialize embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Compute course embeddings
course_embeddings = model.encode(courses_df['description'].tolist(), convert_to_numpy=True)

# Build FAISS index
dimension = course_embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)  # Inner product = cosine similarity if normalized embeddings
faiss.normalize_L2(course_embeddings)  # Normalize for cosine similarity
index.add(course_embeddings)

def recommend_courses(user_query, top_k=5):
    # Embed user query
    query_emb = model.encode([user_query], convert_to_numpy=True)
    faiss.normalize_L2(query_emb)
    
    # Search index
    distances, indices = index.search(query_emb, top_k)
    
    # Fetch recommended courses
    recommended_courses = courses_df.iloc[indices[0]][['title', 'description']].copy()
    recommended_courses['similarity'] = distances[0]
    return recommended_courses

# Example user query
user_query = "I've completed the 'Python Programming for Data Science' course and enjoy data visualization."
recommended = recommend_courses(user_query)
print(recommended)
