#!/usr/bin/env python3

"""
Simple launcher script for the API server.
This makes it easier to run the server with standard options.
"""

import uvicorn
import argparse
import logging
import os

def main():
    parser = argparse.ArgumentParser(description="Run the Document Embedding API server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind the server to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload (for development)")
    parser.add_argument("--log-level", default="info", choices=["debug", "info", "warning", "error", "critical"], 
                       help="Logging level")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = getattr(logging, args.log_level.upper())
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    logger = logging.getLogger("document-embedding-api")
    logger.info(f"Starting Document Embedding API on {args.host}:{args.port}")
    logger.info(f"Log level set to: {args.log_level.upper()}")
    
    if args.reload:
        logger.info("Auto-reload enabled (development mode)")
    
    # Run the API server
    uvicorn.run(
        "main:app", 
        host=args.host, 
        port=args.port, 
        reload=args.reload,
        workers=args.workers,
        log_level=args.log_level.lower()
    )

if __name__ == "__main__":
    main()
