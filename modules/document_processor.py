#!/usr/bin/env python3

from typing import List, Dict, Any
import re
import markdown
from bs4 import BeautifulSoup

def process_markdown(markdown_text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
    """
    Process markdown text into chunks.
    
    Args:
        markdown_text: Markdown formatted text
        chunk_size: Target size of each chunk in characters
        overlap: Number of overlapping characters between chunks
        
    Returns:
        List of text chunks
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
            # Add heading level indicators
            level = int(element.name[1])
            heading = f"{'#' * level} {element.text.strip()}"
            paragraphs.append(heading)
        else:
            paragraphs.append(element.text.strip())
            
    # Now chunk the paragraphs
    return chunk_text(paragraphs, chunk_size, overlap)

def chunk_text(paragraphs: List[str], chunk_size: int = 1000, overlap: int = 100) -> List[str]:
    """
    Chunk text into smaller pieces with overlap.
    
    Args:
        paragraphs: List of paragraphs
        chunk_size: Target size of each chunk in characters
        overlap: Number of overlapping characters between chunks
        
    Returns:
        List of chunks
    """
    chunks = []
    current_chunk = []
    current_length = 0
    
    for paragraph in paragraphs:
        # Skip empty paragraphs
        if not paragraph.strip():
            continue
            
        # If adding paragraph exceeds target chunk size, finalize current chunk
        if current_length + len(paragraph) > chunk_size and current_chunk:
            chunks.append("\n\n".join(current_chunk))
            
            # Keep some content for overlap if needed
            if overlap > 0:
                # Determine how many paragraphs to keep for overlap
                overlap_size = 0
                overlap_paragraphs = []
                
                for p in reversed(current_chunk):
                    if overlap_size + len(p) <= overlap:
                        overlap_paragraphs.insert(0, p)
                        overlap_size += len(p)
                    else:
                        break
                        
                current_chunk = overlap_paragraphs
                current_length = overlap_size
            else:
                current_chunk = []
                current_length = 0
                
        # Add paragraph to current chunk
        current_chunk.append(paragraph)
        current_length += len(paragraph)
    
    # Add the final chunk if it's not empty
    if current_chunk:
        chunks.append("\n\n".join(current_chunk))
        
    return chunks

def subchunk_for_query(text: str, max_length: int = 100) -> List[str]:
    """
    Create smaller subchunks for query comparison to improve matching with specific parts of documents.
    
    Args:
        text: Text to subchunk
        max_length: Maximum length of each subchunk
        
    Returns:
        List of subchunks
    """
    # Split by sentences first
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    subchunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        # If sentence itself is too long, split by comma or other delimiters
        if len(sentence) > max_length:
            # Add current accumulated chunk if not empty
            if current_chunk:
                subchunks.append(" ".join(current_chunk))
                current_chunk = []
                current_length = 0
                
            # Split long sentence
            parts = re.split(r'(?<=[,;:])\s+', sentence)
            for part in parts:
                if len(part) > max_length:
                    # If still too long, just add it as is
                    subchunks.append(part)
                else:
                    if current_length + len(part) > max_length:
                        subchunks.append(" ".join(current_chunk))
                        current_chunk = [part]
                        current_length = len(part)
                    else:
                        current_chunk.append(part)
                        current_length += len(part)
        else:
            # Add sentence if it fits
            if current_length + len(sentence) > max_length:
                subchunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_length = len(sentence)
            else:
                current_chunk.append(sentence)
                current_length += len(sentence)
    
    # Add final chunk
    if current_chunk:
        subchunks.append(" ".join(current_chunk))
        
    return subchunks
