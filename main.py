#!/usr/bin/env python3

import os
import asyncio
import logging
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uuid
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("document-embedding-api")

# Import embedding service and document processor
from modules.embedding_service import generate_embedding, cosine_similarity
from modules.mongodb_config import get_client
from modules.document_processor import process_markdown, chunk_text

# Initialize FastAPI app
app = FastAPI(
    title="Document Embedding API",
    description="API for document embedding and semantic search",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Modify in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models for API
class SearchQuery(BaseModel):
    query: str
    user_id: str
    limit: int = 3
    similarity_threshold: float = 0.5

class SearchResult(BaseModel):
    document_id: str
    chunk_id: str
    content: str
    similarity: float
    metadata: Dict[str, Any] = {}

class DocumentUpload(BaseModel):
    user_id: str
    document_name: str = None
    metadata: Dict[str, Any] = {}

class DocumentResponse(BaseModel):
    document_id: str
    user_id: str
    document_name: str
    chunk_count: int
    created_at: datetime

# Initialize MongoDB client
mongo_client = get_client()
db = mongo_client["embeddings_db"]
user_collection = db["users"]
document_collection = db["documents"]
chunk_collection = db["chunks"]

# Ensure indexes
def setup_db():
    """Create database indexes for better performance"""
    try:
        # Create indexes for better performance
        chunk_collection.create_index([("user_id", 1)])
        chunk_collection.create_index([("document_id", 1)])
        document_collection.create_index([("user_id", 1)])
        logger.info("Database indexes created successfully")
    except Exception as e:
        logger.error(f"Error creating database indexes: {str(e)}")
        logger.warning("Application will continue without database indexing")
        # Don't raise the exception - let the app start without proper DB indexing
        # This allows the embedding model to still function even with DB issues

@app.on_event("startup")
async def startup_event():
    """Run on application startup"""
    logger.info("Starting Document Embedding API")
    try:
        setup_db()
    except Exception as e:
        logger.error(f"Database setup failed: {str(e)}")
        logger.warning("Application will start with limited functionality")

@app.on_event("shutdown")
async def shutdown_event():
    """Run on application shutdown"""
    logger.info("Shutting down Document Embedding API")
    # mongo_client closes automatically via atexit handler in embedding_service

# Endpoints
@app.post("/documents/upload", response_model=DocumentResponse)
async def upload_document(
    file: UploadFile = File(...),
    user_id: str = Form(...),
    document_name: Optional[str] = Form(None),
    metadata: Optional[str] = Form("{}")
):
    """
    Upload a markdown document, process it into chunks, and store embeddings
    """
    try:
        # Read the file content
        content = await file.read()
        content_str = content.decode("utf-8")
        
        # If document_name not provided, use filename
        if not document_name:
            document_name = file.filename
        
        logger.info(f"Processing document: {document_name} for user: {user_id}")
            
        # Process document
        document_id = str(uuid.uuid4())
        chunks = process_markdown(content_str)
        
        # Store document metadata
        doc = {
            "_id": document_id,
            "user_id": user_id,
            "document_name": document_name,
            "original_filename": file.filename,
            "metadata": metadata,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        
        document_collection.insert_one(doc)
        
        # Process and store chunks with embeddings
        chunk_ids = []
        for i, chunk in enumerate(chunks):
            chunk_id = f"{document_id}_{i}"
            
            # Generate embedding
            embedding = await generate_embedding(chunk)
            
            if embedding:
                # Store chunk with embedding
                chunk_doc = {
                    "_id": chunk_id,
                    "document_id": document_id,
                    "user_id": user_id,
                    "chunk_index": i,
                    "content": chunk,
                    "embedding": embedding,
                    "vector_size": len(embedding),
                    "created_at": datetime.utcnow()
                }
                
                chunk_collection.insert_one(chunk_doc)
                chunk_ids.append(chunk_id)
            
        # Update document with chunk count
        document_collection.update_one(
            {"_id": document_id},
            {"$set": {"chunk_count": len(chunk_ids)}}
        )
        
        logger.info(f"Document processed successfully. ID: {document_id}, Chunks: {len(chunk_ids)}")
        
        return {
            "document_id": document_id,
            "user_id": user_id,
            "document_name": document_name,
            "chunk_count": len(chunk_ids),
            "created_at": doc["created_at"]
        }
        
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

@app.post("/search", response_model=List[SearchResult])
async def semantic_search(query: SearchQuery):
    """
    Perform semantic search across user's documents
    """
    try:
        logger.info(f"Searching for: '{query.query}' for user: {query.user_id}")
        
        # Generate embedding for query
        query_embedding = await generate_embedding(query.query)
        
        if not query_embedding:
            logger.error("Failed to generate embedding for query")
            raise HTTPException(status_code=400, detail="Failed to generate embedding for query")
            
        # Fetch all chunks for this user
        user_chunks = list(chunk_collection.find({"user_id": query.user_id}))
        logger.info(f"Found {len(user_chunks)} chunks for user {query.user_id}")
        
        # Calculate similarity scores
        results = []
        for chunk in user_chunks:
            if "embedding" in chunk:
                similarity = cosine_similarity(query_embedding, chunk["embedding"])
                
                if similarity >= query.similarity_threshold:
                    # Get document info
                    document = document_collection.find_one({"_id": chunk["document_id"]})
                    
                    result = {
                        "document_id": chunk["document_id"],
                        "chunk_id": chunk["_id"],
                        "content": chunk["content"],
                        "similarity": similarity,
                        "metadata": {
                            "document_name": document.get("document_name", "Unknown") if document else "Unknown",
                            "chunk_index": chunk.get("chunk_index", 0),
                        }
                    }
                    results.append(result)
        
        # Sort by similarity (highest first)
        results = sorted(results, key=lambda x: x["similarity"], reverse=True)
        
        # Return top results
        top_results = results[:query.limit]
        logger.info(f"Returning {len(top_results)} results for query: '{query.query}'")
        return top_results
        
    except Exception as e:
        logger.error(f"Error during search: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error during search: {str(e)}")

@app.get("/documents", response_model=List[Dict[str, Any]])
async def get_user_documents(user_id: str):
    """Get all documents for a specific user"""
    try:
        logger.info(f"Retrieving documents for user: {user_id}")
        documents = list(document_collection.find({"user_id": user_id}))
        # Convert ObjectId to string for JSON serialization
        for doc in documents:
            doc["_id"] = str(doc["_id"])
        logger.info(f"Found {len(documents)} documents for user: {user_id}")
        return documents
    except Exception as e:
        logger.error(f"Error retrieving documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving documents: {str(e)}")

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    return {"status": "ok", "service": "document-embedding-api"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
