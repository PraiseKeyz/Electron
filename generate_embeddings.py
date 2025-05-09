# import requests
# import json
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv


load_dotenv()

model = SentenceTransformer("all-MiniLM-L6-v2")

MONGO_URI = os.getenv("MONGO_URI")
# MongoDB Connection
client = MongoClient(MONGO_URI)
db = client["chemistry_db"]
chunks_collection = db["course_chunks"]  # Collection where text is stored
vector_collection = db["vector_store"]   # Collection to store embeddings


# Function to get embeddings
def get_embedding(text):
    return model.encode(text).tolist()

# Process all stored chunks
def generate_and_store_embeddings():
    for chunk in chunks_collection.find():  # Fetch all stored chunks
        text = chunk["text"]
        course_name = chunk["course_name"]

        # Get embedding
        embedding = get_embedding(text)
        if embedding:
            # Save embedding in MongoDB
            vector_collection.insert_one({
                "course": course_name,
                "text": text,
                "embeddings": embedding  # Extract first embedding
            })
            print(f"{embedding}")

# Run the embedding process
generate_and_store_embeddings()
print("🚀 All embeddings stored successfully!")
