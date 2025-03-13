#!/usr/bin/env python3

import requests
import json
import sys
import os
from typing import List, Dict, Any
import argparse

# API endpoint
API_BASE_URL = "http://localhost:8000"

def upload_document(user_id: str, file_path: str, document_name: str = None) -> Dict[str, Any]:
    """Upload a document to the API"""
    url = f"{API_BASE_URL}/documents/upload"
    
    # Prepare file upload
    files = {"file": open(file_path, "rb")}
    
    # If no document name provided, use filename
    if not document_name:
        document_name = os.path.basename(file_path)
    
    # Prepare data
    data = {
        "user_id": user_id,
        "document_name": document_name,
        "metadata": json.dumps({"local_path": file_path})
    }
    
    # Send request
    response = requests.post(url, files=files, data=data)
    
    if response.status_code != 200:
        print(f"Error uploading document: {response.text}")
        return None
        
    return response.json()

def search_documents(user_id: str, query: str, limit: int = 3) -> List[Dict[str, Any]]:
    """Search documents by semantic similarity"""
    url = f"{API_BASE_URL}/search"
    
    # Prepare data
    data = {
        "user_id": user_id,
        "query": query,
        "limit": limit,
        "similarity_threshold": 0.5
    }
    
    # Send request
    response = requests.post(url, json=data)
    
    if response.status_code != 200:
        print(f"Error searching documents: {response.text}")
        return []
        
    return response.json()

def list_documents(user_id: str) -> List[Dict[str, Any]]:
    """List all documents for a user"""
    url = f"{API_BASE_URL}/documents?user_id={user_id}"
    
    # Send request
    response = requests.get(url)
    
    if response.status_code != 200:
        print(f"Error listing documents: {response.text}")
        return []
        
    return response.json()

def main():
    parser = argparse.ArgumentParser(description="Test client for the Document Embedding API")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Upload command
    upload_parser = subparsers.add_parser("upload", help="Upload a document")
    upload_parser.add_argument("--user", "-u", required=True, help="User ID")
    upload_parser.add_argument("--file", "-f", required=True, help="File path")
    upload_parser.add_argument("--name", "-n", help="Document name (optional)")
    
    # Search command
    search_parser = subparsers.add_parser("search", help="Search documents")
    search_parser.add_argument("--user", "-u", required=True, help="User ID")
    search_parser.add_argument("--query", "-q", required=True, help="Search query")
    search_parser.add_argument("--limit", "-l", type=int, default=3, help="Result limit")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List documents")
    list_parser.add_argument("--user", "-u", required=True, help="User ID")
    
    args = parser.parse_args()
    
    if args.command == "upload":
        result = upload_document(args.user, args.file, args.name)
        if result:
            print(f"Document uploaded successfully. Document ID: {result['document_id']}")
            print(f"Chunks created: {result['chunk_count']}")
            
    elif args.command == "search":
        results = search_documents(args.user, args.query, args.limit)
        if results:
            print(f"\nFound {len(results)} results:")
            for i, result in enumerate(results, 1):
                print(f"{i}. Similarity: {result['similarity']:.4f}")
                print(f"   Document: {result['metadata'].get('document_name', 'Unknown')}")
                print(f"   Content: {result['content'][:200]}...\n")
        else:
            print("No results found.")
            
    elif args.command == "list":
        documents = list_documents(args.user)
        if documents:
            print(f"\nFound {len(documents)} documents:")
            for i, doc in enumerate(documents, 1):
                print(f"{i}. {doc['document_name']} (ID: {doc['_id']})")
                print(f"   Chunks: {doc.get('chunk_count', 'Unknown')}")
                print(f"   Created: {doc.get('created_at', 'Unknown')}\n")
        else:
            print("No documents found.")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
