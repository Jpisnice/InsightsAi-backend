#!/usr/bin/env python3

import asyncio
import sys
import os
import logging
from typing import List, Dict, Any, Union, Optional
from datetime import datetime
import numpy as np
import importlib
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
import atexit
import pinecone
from pinecone import Pinecone, ServerlessSpec
import pathlib
from dotenv import load_dotenv  # Add explicit import for dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("embedding-service")

# Be more selective about which modules to clear from cache
# IMPORTANT: Don't clear the current module or we'll break imports
logger.info("Selectively clearing module cache")
for module_name in list(sys.modules.keys()):
    if module_name.startswith('modules.') and module_name != 'modules.embedding_service':
        logger.info(f"Removing {module_name} from sys.modules")
        sys.modules.pop(module_name, None)

# Explicitly load environment variables from .env file
env_path = pathlib.Path(__file__).parent.parent / '.env'
logger.info(f"Loading fresh environment variables from: {env_path}")
load_dotenv(dotenv_path=env_path, override=True)  # Force override of existing variables

# Import MongoDB config - after loading env vars to ensure it has access
from modules.mongodb_config import get_client

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

# Initialize Pinecone with straight index name from env file 
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.environ.get("PINECONE_ENV", "us-east-1")

# Direct reading of .env file for PINECONE_INDEX to ensure we get the correct value
try:
    with open(env_path, 'r') as env_file:
        for line in env_file:
            line = line.strip()
            if line and not line.startswith('#') and 'PINECONE_INDEX=' in line:
                # Extract value directly from file
                raw_index_name = line.split('PINECONE_INDEX=')[1].strip()
                if raw_index_name:
                    logger.info(f"Extracted index name directly from .env file: {raw_index_name}")
                    break
        else:
            # If no PINECONE_INDEX found in file, use environment variable
            raw_index_name = os.environ.get("PINECONE_INDEX")
except Exception as e:
    logger.warning(f"Error reading .env file directly: {e}")
    # Fallback to environment variable
    raw_index_name = os.environ.get("PINECONE_INDEX")

logger.info(f"Raw index name: {raw_index_name}")

# Clean up the raw index name - strip any inline comments (anything after #)
if raw_index_name and '#' in raw_index_name:
    clean_index_name = raw_index_name.split('#')[0].strip()
    logger.info(f"Cleaned index name (removed comments): '{clean_index_name}'")
else:
    clean_index_name = raw_index_name

# If the cleaned name is empty or None, use the default
PINECONE_INDEX = clean_index_name if clean_index_name else "docemb"

# Sanitize index name - ensure it only contains lowercase alphanumeric chars or hyphens
PINECONE_INDEX = ''.join(c for c in PINECONE_INDEX if c.isalnum() or c == '-').lower()

# Log Pinecone configuration for debugging
logger.info(f"Pinecone Configuration:")
logger.info(f"  - API Key: {'*' * 5}{PINECONE_API_KEY[-5:] if PINECONE_API_KEY else 'Not set'}")
logger.info(f"  - Environment: {PINECONE_ENVIRONMENT}")
logger.info(f"  - Final Index Name: '{PINECONE_INDEX}'")

# Initialize variables for Pinecone
pc = None
index = None
pinecone_available = False

# Initialize Pinecone client with better error handling
try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    # Check if index exists, if not try to create it
    try:
        index = pc.Index(PINECONE_INDEX)
        logger.info(f"Connected to Pinecone index: {PINECONE_INDEX}")
        pinecone_available = True
    except Exception as e:
        logger.warning(f"Index {PINECONE_INDEX} not found. Will attempt to create it.")
        try:
            # Create the index with appropriate settings for the model
            pc.create_index(
                name=PINECONE_INDEX,
                dimension=384,  # MiniLM-L6 uses 384 dimensions
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region=PINECONE_ENVIRONMENT)
            )
            logger.info(f"Created new Pinecone index: {PINECONE_INDEX}")
            # Wait briefly for index to initialize
            import time
            time.sleep(5)
            index = pc.Index(PINECONE_INDEX)
            pinecone_available = True
        except Exception as create_error:
            logger.error(f"Failed to create Pinecone index: {create_error}")
            logger.warning("Application will run without Pinecone vector storage.")
except Exception as e:
    logger.error(f"Error connecting to Pinecone: {e}")
    logger.warning("Application will run without Pinecone vector storage.")

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

# Enhanced search that respects hierarchy and uses direct Pinecone filtering
async def hierarchical_search(user_id: str, query: str, folder_id: Optional[str] = None, limit: int = 3, threshold: float = 0.5):
    """
    Search with awareness of document hierarchy using direct Pinecone filtering
    
    Args:
        user_id: ID of the user who owns the documents
        query: Search query text
        folder_id: Optional folder ID to limit search scope
        limit: Maximum number of results to return
        threshold: Minimum similarity score threshold
        
    Returns:
        Hierarchical results grouped by document and chunks
    """
    query_embedding = await generate_embedding(query)
    
    if not query_embedding or not pinecone_available:
        return {}
    
    # Build filter based on user_id and optional folder_id
    filter_dict = {"user_id": {"$eq": user_id}, "type": {"$eq": "subchunk"}}
    
    if folder_id:
        filter_dict["folder_id"] = {"$eq": folder_id}
        
    # First try to match at subchunk level for precision
    results = index.query(
        vector=query_embedding,
        top_k=limit*3,  # Get more results to consolidate
        include_metadata=True,
        filter=filter_dict
    )
    
    # Group results by document and chunk for hierarchical presentation
    hierarchical_results = {}
    for match in results["matches"]:
        if match["score"] < threshold:
            continue
            
        doc_id = match["metadata"]["document_id"]
        chunk_id = match["metadata"]["chunk_id"]
        
        if doc_id not in hierarchical_results:
            # Fetch document details from MongoDB
            document = mongo_client["embeddings_db"]["documents"].find_one({"_id": doc_id})
            if document:
                hierarchical_results[doc_id] = {
                    "document_name": document.get("document_name", "Unknown"),
                    "folder_id": document.get("folder_id", "Unknown"),
                    "chunks": {}
                }
            else:
                hierarchical_results[doc_id] = {"document_name": "Unknown", "chunks": {}}
            
        if chunk_id not in hierarchical_results[doc_id]["chunks"]:
            hierarchical_results[doc_id]["chunks"][chunk_id] = []
            
        hierarchical_results[doc_id]["chunks"][chunk_id].append({
            "content": match["metadata"]["content"],
            "similarity": match["score"]
        })
    
    # Return ordered by document relevance (calculate average similarity per document)
    doc_relevance = {}
    for doc_id, doc_data in hierarchical_results.items():
        similarities = []
        for chunk_matches in doc_data["chunks"].values():
            for match in chunk_matches:
                similarities.append(match["similarity"])
        doc_relevance[doc_id] = sum(similarities) / len(similarities) if similarities else 0
    
    # Sort documents by relevance
    sorted_results = {k: hierarchical_results[k] for k in sorted(doc_relevance.keys(), key=lambda x: doc_relevance[x], reverse=True)}
    
    return sorted_results

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
