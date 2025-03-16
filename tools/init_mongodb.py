#!/usr/bin/env python3

import os
import sys
import asyncio
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

# Add parent directory to Python path so we can import modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

# Now we can import from modules
from modules.mongodb_config import get_client, setup_collections, create_indexes

async def main():
    """Initialize MongoDB collections and indexes."""
    print("Initializing MongoDB collections and indexes...")
    
    try:
        # Get MongoDB client and test connection
        client = get_client()
        # Check if we can connect to MongoDB
        client.admin.command('ping')
        print("Connected to MongoDB successfully.")
        
        # Setup collections
        print("Setting up collections...")
        collections = setup_collections()
        print(f"Collections created: {collections}")
        
        # Create indexes for better performance
        print("Creating indexes...")
        create_indexes()
        
        print("MongoDB setup complete!")
        
        # Additional information
        db_names = client.list_database_names()
        print(f"Available databases: {db_names}")
        
        # List collections in each database
        for db_name in db_names:
            if db_name not in ['admin', 'config', 'local']:
                db = client[db_name]
                collections = db.list_collection_names()
                print(f"Collections in '{db_name}': {collections}")
    
    except ConnectionFailure:
        print("ERROR: Failed to connect to MongoDB. Is the MongoDB service running?")
        print("Try running: sudo systemctl start mongod")
        sys.exit(1)
    except ServerSelectionTimeoutError:
        print("ERROR: MongoDB server selection timeout. Check if MongoDB is running and accessible.")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: An unexpected error occurred: {e}")
        sys.exit(1)
    finally:
        # Close connection
        if 'client' in locals():
            client.close()
            print("MongoDB connection closed.")

if __name__ == "__main__":
    asyncio.run(main())
