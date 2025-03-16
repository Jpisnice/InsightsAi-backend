#!/usr/bin/env python3

import os
import asyncio
import logging
import json
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
from modules.embedding_service import generate_embedding, cosine_similarity, index, pinecone_available
from modules.mongodb_config import get_client
from modules.document_processor import process_markdown, chunk_text, subchunk_for_query

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
    folder_id: Optional[str] = None
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
    folder_id: str
    document_name: Optional[str] = None
    metadata: Dict[str, Any] = {}

class FolderCreate(BaseModel):
    user_id: str
    folder_name: str
    description: Optional[str] = ""

class DocumentResponse(BaseModel):
    document_id: str
    user_id: str
    folder_id: str
    document_name: str
    chunk_count: int
    created_at: datetime

class FolderResponse(BaseModel):
    folder_id: str
    user_id: str
    folder_name: str
    description: str
    document_count: int = 0
    created_at: datetime

# Initialize MongoDB client
mongo_client = get_client()
db = mongo_client["embeddings_db"]
user_collection = db["users"]
folder_collection = db["folders"]
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
        document_collection.create_index([("folder_id", 1)])
        folder_collection.create_index([("user_id", 1)])
        logger.info("Database indexes created successfully")
    except Exception as e:
        logger.error(f"Error creating database indexes: {str(e)}")
        logger.warning("Application will continue without database indexing")

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

# New folder management endpoints
@app.post("/folders", response_model=FolderResponse)
async def create_folder(folder: FolderCreate):
    """Create a new folder for organizing documents"""
    try:
        folder_id = str(uuid.uuid4())
        now = datetime.utcnow()
        
        folder_doc = {
            "_id": folder_id,
            "user_id": folder.user_id,
            "folder_name": folder.folder_name,
            "description": folder.description,
            "document_count": 0,
            "created_at": now,
            "updated_at": now
        }
        
        folder_collection.insert_one(folder_doc)
        logger.info(f"Created new folder '{folder.folder_name}' for user {folder.user_id}")
        
        # Map _id to folder_id to match the response model
        response_doc = dict(folder_doc)
        response_doc["folder_id"] = response_doc.pop("_id")
        
        return response_doc
    except Exception as e:
        logger.error(f"Error creating folder: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating folder: {str(e)}")

@app.get("/folders", response_model=List[FolderResponse])
async def get_user_folders(user_id: str):
    """Get all folders for a specific user"""
    try:
        folders = list(folder_collection.find({"user_id": user_id}))
        # Convert ObjectId to string for JSON serialization and map _id to folder_id
        for folder in folders:
            if isinstance(folder.get("_id"), object):
                folder["folder_id"] = str(folder.pop("_id"))
            else:
                folder["folder_id"] = folder.pop("_id")
        
        logger.info(f"Retrieved {len(folders)} folders for user {user_id}")
        return folders
    except Exception as e:
        logger.error(f"Error retrieving folders: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving folders: {str(e)}")

@app.get("/folders/{folder_id}/documents", response_model=List[Dict[str, Any]])
async def get_folder_documents(folder_id: str):
    """Get all documents in a specific folder"""
    try:
        documents = list(document_collection.find({"folder_id": folder_id}))
        # Convert ObjectId to string for JSON serialization
        for doc in documents:
            if isinstance(doc.get("_id"), object):
                doc["document_id"] = str(doc.pop("_id"))
            else:
                doc["document_id"] = doc.pop("_id")
        
        logger.info(f"Retrieved {len(documents)} documents for folder {folder_id}")
        return documents
    except Exception as e:
        logger.error(f"Error retrieving folder documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving folder documents: {str(e)}")

# Modified document upload to include folder_id
@app.post("/documents/upload", response_model=DocumentResponse)
async def upload_document(
    file: UploadFile = File(...),
    user_id: str = Form(...),
    folder_id: str = Form(...),
    document_name: Optional[str] = Form(None),
    metadata: Optional[str] = Form("{}")
):
    """
    Upload a markdown document, process it into chunks, and store embeddings in Pinecone
    """
    try:
        # Check if folder exists
        folder = folder_collection.find_one({"_id": folder_id, "user_id": user_id})
        if not folder:
            raise HTTPException(status_code=404, detail=f"Folder {folder_id} not found or doesn't belong to user {user_id}")
        
        # Read the file content
        content = await file.read()
        content_str = content.decode("utf-8")
        
        # If document_name not provided, use filename
        if not document_name:
            document_name = file.filename
        
        logger.info(f"Processing document: {document_name} for user: {user_id} in folder: {folder_id}")
            
        # Process document
        document_id = str(uuid.uuid4())
        chunks = process_markdown(content_str)
        
        # Store document metadata
        doc = {
            "_id": document_id,
            "user_id": user_id,
            "folder_id": folder_id,
            "document_name": document_name,
            "original_filename": file.filename,
            "metadata": metadata,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        
        document_collection.insert_one(doc)
        
        # Update folder's document count
        folder_collection.update_one(
            {"_id": folder_id},
            {"$inc": {"document_count": 1}, "$set": {"updated_at": datetime.utcnow()}}
        )
        
        # Process and store chunks with embeddings in Pinecone
        chunk_ids = []
        for i, chunk in enumerate(chunks):
            chunk_id = f"{document_id}_{i}"
            
            # Generate embedding for chunk
            embedding = await generate_embedding(chunk)
            
            if embedding:
                # Store chunk with embedding in Pinecone
                vector = (chunk_id, embedding, {
                    "document_id": document_id, 
                    "folder_id": folder_id,
                    "user_id": user_id, 
                    "content": chunk, 
                    "type": "chunk"
                })
                index.upsert([vector])
                
                # Process into smaller subchunks for more precise matching
                subchunks = subchunk_for_query(chunk, max_length=100)
                for j, subchunk in enumerate(subchunks):
                    subchunk_id = f"{chunk_id}_sub{j}"
                    subchunk_embedding = await generate_embedding(subchunk)
                    if subchunk_embedding:
                        # Store subchunk with embedding and reference to parent chunk
                        vector = (subchunk_id, subchunk_embedding, {
                            "document_id": document_id, 
                            "folder_id": folder_id,
                            "chunk_id": chunk_id,
                            "user_id": user_id, 
                            "content": subchunk,
                            "type": "subchunk"
                        })
                        index.upsert([vector])
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
            "folder_id": folder_id,
            "document_name": document_name,
            "chunk_count": len(chunk_ids),
            "created_at": doc["created_at"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

# Updated search to use direct Pinecone filtering
@app.post("/search", response_model=List[SearchResult])
async def semantic_search(query: SearchQuery):
    """
    Perform semantic search across user's documents using direct Pinecone filtering
    """
    try:
        logger.info(f"Searching for: '{query.query}' for user: {query.user_id}")
        
        if not pinecone_available:
            raise HTTPException(status_code=503, detail="Pinecone service unavailable")
        
        # Generate embedding for query
        query_embedding = await generate_embedding(query.query)
        
        if not query_embedding:
            logger.error("Failed to generate embedding for query")
            raise HTTPException(status_code=400, detail="Failed to generate embedding for query")
            
        # Build filter based on user_id and optional folder_id
        filter_dict = {"user_id": {"$eq": query.user_id}}
        
        # Add folder filter if provided
        if query.folder_id:
            filter_dict["folder_id"] = {"$eq": query.folder_id}
            
        # Query Pinecone directly with filters
        results = index.query(
            vector=query_embedding,
            top_k=query.limit * 2,  # Get extra results for filtering
            include_metadata=True,
            filter=filter_dict
        )
        
        search_results = []
        for match in results["matches"]:
            # Skip results below threshold
            if match["score"] < query.similarity_threshold:
                continue
                
            chunk_id = match["id"]
            document_id = match["metadata"]["document_id"]
            
            # Retrieve document metadata from MongoDB
            document = document_collection.find_one({"_id": document_id})
            
            if document:
                search_results.append({
                    "document_id": document_id,
                    "chunk_id": chunk_id,
                    "content": match["metadata"]["content"],
                    "similarity": match["score"],
                    "metadata": {
                        "document_name": document.get("document_name", "Unknown"),
                        "folder_id": document.get("folder_id", "Unknown"),
                        "folder_name": folder_collection.find_one({"_id": document.get("folder_id")}).get("folder_name", "Unknown") if document.get("folder_id") else "Unknown",
                        "chunk_type": match["metadata"].get("type", "chunk"),
                    }
                })
        
        # Sort by similarity (highest first)
        search_results = sorted(search_results, key=lambda x: x["similarity"], reverse=True)
        
        # Return top results
        top_results = search_results[:query.limit]
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
