import json
import os
from google.cloud import aiplatform
import pinecone

# Initialize Pinecone
pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment='us-west1-gcp')
index_name = "rag-kb-index"

if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=768)  # Adjust dimension as per embedding size

index = pinecone.Index(index_name)

def load_kb(file_path):
    with open(file_path) as f:
        return json.load(f)

def embed_texts(texts):
    client = aiplatform.gapic.PredictionServiceClient()
    embeddings = []
    for text in texts:
        response = client.embed_text(
            model="models/gemini-embedding-001",
            instances=[{"content": text}],
        )
        embeddings.append(response.predictions[0]["embedding"])
    return embeddings

def index_kb():
    kb = load_kb("self_critique_loop_dataset.json")
    texts = [item["text"] for item in kb]
    ids = [item["id"] for item in kb]
    embeddings = embed_texts(texts)

    vectors = [(ids[i], embeddings[i]) for i in range(len(ids))]
    index.upsert(vectors)
    print(f"Upserted {len(vectors)} vectors to Pinecone.")

if __name__ == "__main__":
    index_kb()
