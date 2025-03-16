import os
import pathlib
from dotenv import load_dotenv
from pymongo import MongoClient, ASCENDING, TEXT
from pymongo.database import Database
from pymongo.collection import Collection
from pymongo.operations import IndexModel
from typing import Dict, Any, List
import logging
import certifi  # new import added

# Configure logging
logger = logging.getLogger(__name__)

# Load environment variables from .env file - use absolute path to ensure it's found
env_path = pathlib.Path(__file__).parents[2] / '.env'
load_dotenv(dotenv_path=env_path)

# MongoDB connection settings - add better debug info
MONGODB_URI = os.environ.get("MONGODB_URI")
if not MONGODB_URI:
    logger.error("MONGODB_URI environment variable not found in .env file!")
    MONGODB_URI = "mongodb://localhost:27017"
else:
    # Log a sanitized version of the URI for debugging
    safe_uri = MONGODB_URI
    if '@' in safe_uri:
        prefix, suffix = safe_uri.split('@', 1)
        if ':' in prefix and '/' in prefix:
            protocol, credentials = prefix.split('://', 1)
            username, password = credentials.split(':', 1)
            safe_uri = f"{protocol}://{username}:****@{suffix}"
    logger.info(f"Using MongoDB URI: {safe_uri}")

DB_NAME = os.getenv("DB_NAME", "embeddings_db")
logger.info(f"Using database: {DB_NAME}")

# Collections
DOCUMENT_COLLECTION = "documents"
EMBEDDING_COLLECTION = "embeddings"

client: MongoClient = None
db: Database = None

# Collection references
documents: Collection = None
embeddings: Collection = None

# Add a variable to track connection state
client_closed = False

def get_client() -> MongoClient:
    """Get the MongoDB client instance."""
    global client, client_closed
    if client is None or client_closed:
        try:
            logger.info("Attempting to connect to MongoDB...")
            client = MongoClient(
                MONGODB_URI,
                tls=True,
                tlsCAFile=certifi.where(),  # Use certifi to supply valid CA certificates
                connectTimeoutMS=30000,
                socketTimeoutMS=30000,
                retryWrites=True
            )
            # Test connection immediately to catch issues early
            client.admin.command('ping')
            logger.info("Successfully connected to MongoDB")
            client_closed = False
        except Exception as e:
            logger.error(f"MongoDB connection error: {str(e)}")
            # Provide a fallback to local MongoDB if available
            if "mongodb://localhost" not in MONGODB_URI:
                logger.info("Attempting fallback to local MongoDB...")
                try:
                    client = MongoClient("mongodb://localhost:27017")
                    client.admin.command('ping')
                    logger.info("Connected to local MongoDB successfully")
                except Exception as local_e:
                    logger.error(f"Local MongoDB connection also failed: {str(local_e)}")
                    raise
            else:
                raise
    return client

def get_database() -> Database:
    """Get the database instance."""
    global db
    if db is None:
        db = get_client()[DB_NAME]
    return db

def get_collection(collection_name: str) -> Collection:
    """Get a specific collection."""
    return get_database()[collection_name]

def setup_collections():
    """
    Create MongoDB collections that align with the schema.prisma structure.
    Returns a list of collections created.
    """
    global documents, embeddings
    client = get_client()
    db = client["embeddings_db"]
    
    # Define collections based on your Prisma schema
    collections = [
        "documents",  # For storing documents with embeddings
        "folders",    # NEW: For organizing documents into folders
        "users",      # If you had a users table
        "chunks",     # For storing document chunks
        # Add other collections based on your schema
    ]
    
    created = []
    for collection_name in collections:
        if collection_name not in db.list_collection_names():
            db.create_collection(collection_name)
            created.append(collection_name)
    
    documents = get_collection(DOCUMENT_COLLECTION)
    embeddings = get_collection(EMBEDDING_COLLECTION)
    
    return created

def create_indexes():
    """Create necessary indexes for efficient querying."""
    client = get_client()
    db = client["embeddings_db"]
    
    # Text indexes for full-text search
    db.documents.create_index([("title", TEXT), ("content", TEXT)])
    
    # Regular indexes for frequent queries
    db.documents.create_index([("createdAt", ASCENDING)])
    db.documents.create_index("title")
    
    # Create vector index for similarity search
    # try:
    #     # For MongoDB Atlas (uses different API than self-hosted MongoDB)
    #     # all-MiniLM-L6-v2 uses 384-dimensional embeddings
    #     db.command({
    #         "createSearchIndex": {
    #             "name": "vector_index",
    #             "definition": {
    #                 "mappings": {
    #                     "dynamic": False,
    #                     "fields": {
    #                         "embedding": {
    #                             "type": "knnVector",
    #                             "dimensions": 384,
    #                             "similarity": "cosine"
    #                         }
    #                     }
    #                 }
    #             }
    #         }
    #     })
    #     print("Vector search index created successfully")
    # except Exception as e:
    #     print(f"Warning: Could not create vector search index: {e}")
    #     print("Note: Vector search requires MongoDB Atlas with Vector Search or MongoDB 7.0+")
    
    # Don't close the client here as it might be needed later

def is_connected() -> bool:
    """Check if the MongoDB client is connected and available."""
    global client, client_closed
    if client is None or client_closed:
        return False
    try:
        # Ping the server to check if connection is alive
        client.admin.command('ping')
        return True
    except Exception:
        return False

def close_connection():
    """Properly close MongoDB connection when done."""
    global client, client_closed
    if client and not client_closed:
        client.close()
        client_closed = True
        print("MongoDB connection closed.")

# Schema migration function to convert from Prisma/PostgreSQL to MongoDB
def migrate_schema():
    """
    Helper function to convert a Prisma schema to MongoDB collections.
    This is a placeholder - you would need to implement the actual migration
    based on your specific schema.
    """
    # Implement your migration logic here if needed
    pass
