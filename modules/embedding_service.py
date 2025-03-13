#!/usr/bin/env python3

import asyncio
import sys
import os
import logging
from typing import List, Dict, Any, Union, Optional
from datetime import datetime
import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
import atexit

# Import MongoDB config
from modules.mongodb_config import get_client

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("embedding-service")

# Initialize the embedding model
MODEL_NAME = "all-MiniLM-L6-v2"  # You can change this to any model you prefer
logger.info(f"Initializing embedding model: {MODEL_NAME}")

try:
    model = SentenceTransformer(MODEL_NAME)
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    raise

# Create a persistent client connection
mongo_client = get_client()

# Register function to close client connection on exit
def close_mongo_client():
    if mongo_client:
        mongo_client.close()
        logger.info("MongoDB connection closed")

atexit.register(close_mongo_client)

async def generate_embedding(text: str) -> Optional[List[float]]:
    """Generate embedding for a given text using the model."""
    try:
        if not text or not text.strip():
            logger.warning("Empty text provided for embedding generation")
            return None
            
        embedding = model.encode(text)
        return embedding.tolist()  # Convert numpy array to list for MongoDB storage
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        return None

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Calculate cosine similarity between two vectors.
    Returns a value between 0 and 1, where 1 means identical vectors.
    """
    try:
        # Convert distance to similarity (cosine returns distance, 1-distance = similarity)
        return 1 - cosine(vec1, vec2)
    except Exception as e:
        logger.error(f"Error calculating cosine similarity: {e}")
        return 0.0

# Interactive testing function
async def interactive_search(user_id: str, query: str, limit: int = 3, threshold: float = 0.5):
    """
    Interactive search function for testing embeddings.
    
    Args:
        user_id: User ID to search documents for
        query: Search query text
        limit: Maximum number of results to return
        threshold: Minimum similarity score threshold
        
    Returns:
        List of matching document chunks
    """
    db = mongo_client["embeddings_db"]
    chunk_collection = db["chunks"]
    document_collection = db["documents"]
    
    # Generate embedding for query
    query_embedding = await generate_embedding(query)
    
    # Find similar documents
    results = []
    if query_embedding:
        user_chunks = list(chunk_collection.find({"user_id": user_id}))
        
        for chunk in user_chunks:
            if "embedding" in chunk:
                similarity = cosine_similarity(query_embedding, chunk["embedding"])
                
                if similarity >= threshold:
                    document = document_collection.find_one({"_id": chunk["document_id"]})
                    doc_name = document.get("document_name", "Unknown") if document else "Unknown"
                    
                    results.append({
                        "similarity": similarity,
                        "document_name": doc_name,
                        "content": chunk["content"]
                    })
        
        results = sorted(results, key=lambda x: x["similarity"], reverse=True)[:limit]
    
    return results

# Example usage (only runs when script is executed directly)
async def main():
    """Interactive search example for testing"""
    try:
        print("\n=== Embedding Service Test Interface ===\n")
        user_id = input("Enter user ID: ")
        
        while True:
            query = input("\nEnter search query (or 'quit' to exit): ")
            if query.lower() == 'quit':
                break
                
            print(f"Searching for: '{query}'")
            results = await interactive_search(user_id, query)
            
            if results:
                print(f"\nFound {len(results)} similar chunks:")
                for i, result in enumerate(results, 1):
                    print(f"{i}. Similarity: {result.get('similarity', 'N/A'):.4f}, Doc: {result.get('document_name', 'Unknown')}")
                    print(f"   Content: {result.get('content', 'No content')[:100]}...")
            else:
                print("No matching chunks found.")
    except KeyboardInterrupt:
        print("\nExiting application...")
    
if __name__ == "__main__":
    asyncio.run(main())
