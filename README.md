# Document Embedding API

This API provides document embedding and semantic search capabilities using the all-MiniLM-L6-v2 model from Sentence Transformers.

## Features

- Upload markdown documents for processing and embedding
- Automatic chunking of documents with configurable sizes
- Semantic search across documents using cosine similarity
- User-specific document management
- RESTful API with FastAPI

## Prerequisites

- Python 3.11 or higher
- MongoDB instance (local or Atlas)
- [UV](https://github.com/astral-sh/uv) - Fast Python package installer and resolver

## Project Structure

```
mini-lm/
│
├── modules/             # Core functionality modules
│   ├── embedding_service.py    # Embedding generation service
│   ├── document_processor.py   # Document chunking and processing
│   ├── mongodb_config.py       # MongoDB configuration
│   └── database.py             # Database operations
│
├── tools/               # Utility scripts
│   ├── init_mongodb.py  # Initialize MongoDB collections and indexes
│   └── test_client.py   # Test client for API interaction
│
├── main.py              # FastAPI application
├── run.py               # Launcher script
├── pyproject.toml       # Project configuration and dependencies
├── uv.lock              # Dependency lock file 
└── .env                 # Environment variables
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

# Create a virtual environment and install dependencies from pyproject.toml
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Create a lockfile if it doesn't exist and sync dependencies
uv lock
uv sync
```

## Dependency Management

To add or modify dependencies:

```bash
# Add a new dependency
uv add <package-name>

# Add a development dependency
uv add --dev <package-name>

# Remove a dependency
uv remove <package-name>

# View the dependency tree
uv tree
```

## Configuration

1. Configure your MongoDB connection in the `.env` file:

```properties
MONGODB_URI="your-mongodb-connection-string"
```

## Setting Up MongoDB

1. Initialize MongoDB collections and indexes:

```bash
uv run tools/init_mongodb.py
```

This script will:
- Connect to your MongoDB instance
- Create necessary collections
- Set up indexes for efficient querying
- Test the connection and provide feedback

## Running the API Server

1. Start the API server:

```bash
# Basic usage
uv run run.py

# With custom settings
uv run run.py --host 127.0.0.1 --port 8000 --reload --log-level debug
```

Available options:
- `--host`: Host to bind the server to (default: 0.0.0.0)
- `--port`: Port to bind the server to (default: 8000)
- `--reload`: Enable auto-reload for development
- `--log-level`: Set logging level (debug, info, warning, error, critical)
- `--workers`: Number of worker processes

## Using the Test Client

The `test_client.py` script allows you to interact with the API from the command line:

```bash
# Upload a document
uv run tools/test_client.py upload --user user123 --file path/to/document.md

# Search documents
uv run tools/test_client.py search --user user123 --query "your search query" --limit 5

# List documents
uv run tools/test_client.py list --user user123
```

## Customization Options

### Document Processing

You can customize document processing by modifying parameters in `modules/document_processor.py`:

- `chunk_size`: Controls the size of document chunks (default: 1000 characters)
- `overlap`: Controls overlap between chunks for better context (default: 100 characters)

Example:
```python
chunks = process_markdown(content_str, chunk_size=500, overlap=50)
```

### Embedding Model

You can change the embedding model in `modules/embedding_service.py`:

```python
# Change from default "all-MiniLM-L6-v2" to another model
MODEL_NAME = "paraphrase-MiniLM-L3-v2"  # Smaller, faster model
# or
MODEL_NAME = "all-mpnet-base-v2"  # More accurate but slower model
```

Available models can be found in the [sentence-transformers documentation](https://www.sbert.net/docs/pretrained_models.html).

### Search Parameters

Customize search behavior by adjusting:

- `similarity_threshold`: Minimum similarity score (0-1) for results
- `limit`: Maximum number of results to return

## API Endpoints

- `POST /documents/upload`: Upload and process documents
- `POST /search`: Perform semantic search
- `GET /documents`: List user documents
- `GET /health`: Health check endpoint

## Troubleshooting

### MongoDB Connection Issues

If you encounter MongoDB connection errors:

1. Verify your connection string in `.env`
2. Check if MongoDB service is running
3. Ensure network connectivity to MongoDB host
4. Check firewall settings
5. Make sure your IP address is whitelisted in the MongoDB Atlas network access settings

### Model Loading Issues

If the embedding model fails to load:

1. Verify internet connectivity (first run downloads the model)
2. Ensure you have sufficient disk space
3. Check for compatible versions of PyTorch and sentence-transformers

