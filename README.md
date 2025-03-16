# Document Embedding API

This API provides document embedding and semantic search capabilities using the all-MiniLM-L6-v2 model from Sentence Transformers and Pinecone for vector storage.

## Features

- Upload markdown documents for processing and embedding
- Automatic chunking of documents with configurable sizes
- Semantic search across documents using cosine similarity (via Pinecone)
- User-specific document management (MongoDB)
- Folder organization for documents
- Hierarchical document representation for better search context
- RESTful API with FastAPI

## Project Structure

```
mini-lm/
│
├── modules/                   # Core functionality modules
│   ├── embedding_service.py   # Embedding generation and Pinecone integration
│   ├── document_processor.py  # Document chunking and processing
│   ├── mongodb_config.py      # MongoDB configuration
│   ├── database.py            # Database operations
│   └── utils/                 # Utility functions
│
├── tools/                     # Utility scripts
│   ├── init_mongodb.py        # Initialize MongoDB collections and indexes
│   └── test_client.py         # Test client for API interaction
│
├── main.py                    # FastAPI application
├── run.py                     # Launcher script
├── pyproject.toml             # Project configuration and dependencies
└── .env                       # Environment variables
```

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd mini-lm
```

2. Install dependencies using UV:
```bash
# Install UV if you don't have it already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create a virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv sync
```

## Configuration

1. Configure your MongoDB connection and Pinecone API key in the `.env` file:

```properties
MONGODB_URI="your-mongodb-connection-string"
DB_NAME="embeddings_db"
PINECONE_API_KEY="your-pinecone-api-key"
PINECONE_ENV="your-pinecone-environment"  # e.g., us-east-1
PINECONE_INDEX="your-pinecone-index-name"
```

## MongoDB Setup

Initialize MongoDB collections and indexes:

```bash
uv run tools/init_mongodb.py
```

This script will:
- Connect to your MongoDB database
- Create necessary collections (`documents`, `folders`, `users`, `chunks`)
- Set up indexes for efficient querying
- Test the connection and provide feedback

### Database Schema

The project uses the following MongoDB collections:

1. **documents**: Contains document metadata
   ```javascript
   {
     "_id": "document_uuid",
     "user_id": "user_id",
     "folder_id": "folder_id",
     "document_name": "Document Title",
     "content": "Full document content",
     "metadata": { /* Additional metadata */ },
     "created_at": ISODate("2023-01-01T00:00:00Z"),
     "updated_at": ISODate("2023-01-01T00:00:00Z"),
     "chunk_count": 10
   }
   ```

2. **chunks**: Contains document chunks with embeddings
   ```javascript
   {
     "_id": "chunk_id",
     "document_id": "document_uuid",
     "user_id": "user_id",
     "chunk_index": 0,
     "content": "Chunk text content",
     "embedding": [0.1, 0.2, ...],
     "vector_size": 384,
     "created_at": ISODate("2023-01-01T00:00:00Z")
   }
   ```

3. **folders**: Organizes documents into folders
   ```javascript
   {
     "_id": "folder_uuid",
     "user_id": "user_id",
     "folder_name": "Folder Name",
     "description": "Folder description",
     "document_count": 5,
     "created_at": ISODate("2023-01-01T00:00:00Z"),
     "updated_at": ISODate("2023-01-01T00:00:00Z")
   }
   ```

## Running the API Server

Start the API server:

```bash
# Basic usage
uv run run.py

# With custom settings
uv run run.py --host 127.0.0.1 --port 8000 --reload --log-level debug --workers 4
```

Available options:
- `--host`: Host to bind the server to (default: 0.0.0.0)
- `--port`: Port to bind the server to (default: 8000)
- `--reload`: Enable auto-reload for development
- `--log-level`: Set logging level (debug, info, warning, error, critical)
- `--workers`: Number of worker processes

## API Endpoints

### Folder Management
- `POST /folders`: Create a new folder
- `GET /folders`: List user folders
- `GET /folders/{folder_id}/documents`: List documents in a folder

### Document Management
- `POST /documents/upload`: Upload and process documents
- `GET /documents`: List user documents

### Search
- `POST /search`: Perform semantic search

### System
- `GET /health`: Health check endpoint

## Key Code Components

### Document Processing

The document processing system splits documents into chunks for better semantic search:

```python
# From document_processor.py
def process_markdown(markdown_text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
    """
    Process markdown text into chunks.
    """
    # Convert markdown to HTML
    html = markdown.markdown(markdown_text)
    
    # Parse HTML
    soup = BeautifulSoup(html, 'html.parser')
    
    # Extract text from HTML, preserving paragraph structure
    paragraphs = []
    for element in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'li']):
        # Keep headings with special markup
        if element.name.startswith('h'):
            level = int(element.name[1])
            heading = f"{'#' * level} {element.text.strip()}"
            paragraphs.append(heading)
        else:
            paragraphs.append(element.text.strip())
            
    # Now chunk the paragraphs
    return chunk_text(paragraphs, chunk_size, overlap)
```

### Embedding Generation

The embedding service uses Sentence Transformers to create vector representations of text:

```python
# From embedding_service.py
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
```

### Pinecone Integration

The system automatically creates and configures Pinecone indexes:

```python
# From embedding_service.py
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
```

### Semantic Search

The search function uses Pinecone for efficient similarity search:

```python
# From main.py
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
        
        # Process and return results...
    except Exception as e:
        logger.error(f"Error during search: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error during search: {str(e)}")
```

### Hierarchical Document Organization

The system supports organizing documents in folders:

```python
# From main.py
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
        
        # Return folder information...
    except Exception as e:
        logger.error(f"Error creating folder: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating folder: {str(e)}")
```

## Advanced Configuration

### Environment Variable Handling

The system includes robust environment variable management:

```python
# From embedding_service.py
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
```

### Index Name Sanitization

For Pinecone, index names are sanitized to ensure compatibility:

```python
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
```

## Using the Test Client

The `test_client.py` script allows you to interact with the API from the command line:

```bash
# Upload a document
uv run tools/test_client.py upload --user user123 --file path/to/document.md --folder folder_uuid

# Search documents
uv run tools/test_client.py search --user user123 --query "your search query" --limit 5

# List documents
uv run tools/test_client.py list --user user123
```

## Troubleshooting

### MongoDB Connection Issues

If you encounter MongoDB connection errors:

1. Verify your connection string in `.env`
2. Check if MongoDB service is running
3. Ensure network connectivity to MongoDB host
4. Check firewall settings
5. Make sure your IP address is whitelisted in the MongoDB Atlas network access settings

The system includes fallback logic to local MongoDB if remote connection fails:

```python
# From mongodb_config.py
try:
    logger.info("Attempting to connect to MongoDB...")
    client = MongoClient(
        MONGODB_URI,
        tls=True,
        tlsCAFile=certifi.where(),
        connectTimeoutMS=30000,
        socketTimeoutMS=30000,
        retryWrites=True
    )
    # Test connection immediately
    client.admin.command('ping')
    logger.info("Successfully connected to MongoDB")
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
```

### Pinecone Connection Issues

If you encounter Pinecone connection errors:

1. Verify your API key and environment in `.env`
2. Check if the Pinecone index exists
3. Ensure network connectivity to Pinecone

### Model Loading Issues

If the embedding model fails to load:

1. Verify internet connectivity (first run downloads the model)
2. Ensure you have sufficient disk space
3. Check for compatible versions of PyTorch and sentence-transformers

## Performance Considerations

For large document collections, consider:

1. Increasing chunk size for fewer but larger chunks (better context)
2. Decreasing chunk size for more but smaller chunks (more precise matches)
3. Using subchunks for more precise matching within larger contexts
4. Adjusting the similarity threshold for broader or narrower search results

