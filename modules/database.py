#!/usr/bin/env python3

"""
Database utility functions for document and embedding management.
This centralizes database operations for easier maintenance.
"""

import logging
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import uuid
from pymongo import MongoClient
from bson.objectid import ObjectId

from .mongodb_config import get_client

logger = logging.getLogger("database")

# Initialize MongoDB client
mongo_client = get_client()
db = mongo_client["embeddings_db"]
user_collection = db["users"]
document_collection = db["documents"]
chunk_collection = db["chunks"]
folder_collection = db["folders"]

def setup_indexes():
    """Create database indexes for better performance"""
    try:
        chunk_collection.create_index([("user_id", 1)])
        chunk_collection.create_index([("document_id", 1)])
        document_collection.create_index([("user_id", 1)])
        logger.info("Database indexes created successfully")
        return True
    except Exception as e:
        logger.error(f"Error creating database indexes: {str(e)}")
        return False

async def create_folder(
    user_id: str,
    folder_name: str,
    description: str = ""
) -> Optional[str]:
    """
    Create a new folder for a user.
    
    Args:
        user_id: ID of the user who owns the folder
        folder_name: Name of the folder
        description: Optional description of the folder
        
    Returns:
        Folder ID if successful, None otherwise
    """
    try:
        folder_id = str(uuid.uuid4())
        folder_doc = {
            "_id": folder_id,
            "user_id": user_id,
            "folder_name": folder_name,
            "description": description,
            "document_count": 0,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        
        folder_collection.insert_one(folder_doc)
        logger.info(f"Folder created successfully. ID: {folder_id}")
        return folder_id
    except Exception as e:
        logger.error(f"Error creating folder: {str(e)}")
        return None

async def get_user_folders(user_id: str) -> List[Dict[str, Any]]:
    """
    Get all folders for a user.
    
    Args:
        user_id: ID of the user
        
    Returns:
        List of folder objects
    """
    try:
        folders = list(folder_collection.find({"user_id": user_id}))
        # Convert ObjectId to string for JSON serialization
        for folder in folders:
            if isinstance(folder.get("_id"), ObjectId):
                folder["_id"] = str(folder["_id"])
        return folders
    except Exception as e:
        logger.error(f"Error retrieving user folders: {str(e)}")
        return []

async def get_folder_documents(folder_id: str) -> List[Dict[str, Any]]:
    """
    Get all documents in a folder.
    
    Args:
        folder_id: ID of the folder
        
    Returns:
        List of document objects
    """
    try:
        documents = list(document_collection.find({"folder_id": folder_id}))
        # Convert ObjectId to string for JSON serialization
        for doc in documents:
            if isinstance(doc.get("_id"), ObjectId):
                doc["_id"] = str(doc["_id"])
        return documents
    except Exception as e:
        logger.error(f"Error retrieving folder documents: {str(e)}")
        return []

async def store_document(
    user_id: str,
    folder_id: str,
    document_name: str,
    content: str,
    metadata: Dict[str, Any] = None
) -> Optional[str]:
    """
    Store a document in the database.
    
    Args:
        user_id: ID of the user who owns the document
        folder_id: ID of the folder containing the document
        document_name: Name of the document
        content: Document content
        metadata: Additional document metadata
        
    Returns:
        Document ID if successful, None otherwise
    """
    try:
        document_id = str(uuid.uuid4())
        doc = {
            "_id": document_id,
            "user_id": user_id,
            "folder_id": folder_id,
            "document_name": document_name,
            "content": content,
            "metadata": metadata or {},
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        
        document_collection.insert_one(doc)
        
        # Update folder document count
        folder_collection.update_one(
            {"_id": folder_id},
            {"$inc": {"document_count": 1}}
        )
        
        logger.info(f"Document stored successfully. ID: {document_id}")
        return document_id
    except Exception as e:
        logger.error(f"Error storing document: {str(e)}")
        return None

async def store_document_chunk(
    document_id: str,
    user_id: str,
    chunk_index: int,
    content: str,
    embedding: List[float]
) -> Optional[str]:
    """
    Store a document chunk with its embedding.
    
    Args:
        document_id: ID of the parent document
        user_id: ID of the user who owns the document
        chunk_index: Index of this chunk in the document
        content: Chunk content
        embedding: Vector embedding of the chunk
        
    Returns:
        Chunk ID if successful, None otherwise
    """
    try:
        chunk_id = f"{document_id}_{chunk_index}"
        chunk_doc = {
            "_id": chunk_id,
            "document_id": document_id,
            "user_id": user_id,
            "chunk_index": chunk_index,
            "content": content,
            "embedding": embedding,
            "vector_size": len(embedding),
            "created_at": datetime.utcnow()
        }
        
        chunk_collection.insert_one(chunk_doc)
        return chunk_id
    except Exception as e:
        logger.error(f"Error storing document chunk: {str(e)}")
        return None

async def update_document_chunk_count(document_id: str, chunk_count: int) -> bool:
    """
    Update the chunk count for a document.
    
    Args:
        document_id: ID of the document
        chunk_count: Number of chunks
        
    Returns:
        True if successful, False otherwise
    """
    try:
        document_collection.update_one(
            {"_id": document_id},
            {"$set": {"chunk_count": chunk_count, "updated_at": datetime.utcnow()}}
        )
        return True
    except Exception as e:
        logger.error(f"Error updating document chunk count: {str(e)}")
        return False

async def get_user_documents(user_id: str) -> List[Dict[str, Any]]:
    """
    Get all documents for a user.
    
    Args:
        user_id: ID of the user
        
    Returns:
        List of document objects
    """
    try:
        documents = list(document_collection.find({"user_id": user_id}))
        # Convert ObjectId to string for JSON serialization
        for doc in documents:
            if isinstance(doc.get("_id"), ObjectId):
                doc["_id"] = str(doc["_id"])
        return documents
    except Exception as e:
        logger.error(f"Error retrieving user documents: {str(e)}")
        return []

async def get_user_chunks(user_id: str) -> List[Dict[str, Any]]:
    """
    Get all chunks for a user.
    
    Args:
        user_id: ID of the user
        
    Returns:
        List of chunk objects
    """
    try:
        return list(chunk_collection.find({"user_id": user_id}))
    except Exception as e:
        logger.error(f"Error retrieving user chunks: {str(e)}")
        return []

async def get_document_chunks(document_id: str) -> List[Dict[str, Any]]:
    """
    Get all chunks for a document.
    
    Args:
        document_id: ID of the document
        
    Returns:
        List of chunk objects
    """
    try:
        return list(chunk_collection.find({"document_id": document_id}))
    except Exception as e:
        logger.error(f"Error retrieving document chunks: {str(e)}")
        return []

async def delete_document(document_id: str) -> bool:
    """
    Delete a document and all its chunks.
    
    Args:
        document_id: ID of the document
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Delete document
        document_collection.delete_one({"_id": document_id})
        
        # Delete all chunks for this document
        chunk_collection.delete_many({"document_id": document_id})
        
        logger.info(f"Document {document_id} and its chunks deleted successfully")
        return True
    except Exception as e:
        logger.error(f"Error deleting document: {str(e)}")
        return False

# Initialize database indexes when module is imported
setup_indexes()
