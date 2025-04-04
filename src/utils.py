"""
Utility functions for the Multimodal RAG system.
"""
import os
import time
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse

from loguru import logger
from config.config import LOGGING_SETTINGS, LOGS_DIR

# Set up logging
log_file = LOGS_DIR / f"multimodal_rag_{time.strftime('%Y%m%d')}.log"
logger.remove()  # Remove default handler
logger.add(
    sink=log_file,
    level=LOGGING_SETTINGS["level"],
    format=LOGGING_SETTINGS["format"],
    rotation=LOGGING_SETTINGS["rotation"],
    retention=LOGGING_SETTINGS["retention"],
)
logger.add(sink=lambda msg: print(msg), level=LOGGING_SETTINGS["level"])

def get_domain_from_url(url: str) -> str:
    """Extract domain from URL.
    
    Args:
        url: The URL to extract domain from
        
    Returns:
        The domain name
    """
    parsed_url = urlparse(url)
    return parsed_url.netloc

def get_clean_filename(url: str) -> str:
    """Generate a clean filename from URL.
    
    Args:
        url: The URL to generate filename from
        
    Returns:
        A sanitized filename based on the URL
    """
    domain = get_domain_from_url(url)
    path = urlparse(url).path
    
    # Remove inappropriate chars and create a readable filename
    clean_path = path.replace("/", "_").strip("_")
    if not clean_path:
        clean_path = "home"
    
    # If filename is too long, hash part of it
    if len(f"{domain}_{clean_path}") > 100:
        path_hash = hashlib.md5(clean_path.encode()).hexdigest()[:10]
        return f"{domain}_{path_hash}"
    
    return f"{domain}_{clean_path}"

def ensure_dir(directory: Union[str, Path]) -> Path:
    """Ensure directory exists, create if it doesn't.
    
    Args:
        directory: Directory path to ensure
        
    Returns:
        Path object of the directory
    """
    dir_path = Path(directory)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path

def batch_items(items: List[Any], batch_size: int) -> List[List[Any]]:
    """Split a list into batches.
    
    Args:
        items: List of items to batch
        batch_size: Size of each batch
        
    Returns:
        List of batches
    """
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]

def format_exception(e: Exception) -> str:
    """Format exception for logging.
    
    Args:
        e: Exception to format
        
    Returns:
        Formatted exception message
    """
    return f"{type(e).__name__}: {str(e)}"

def safe_request(func):
    """Decorator for safely making requests with retries and error handling.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    from functools import wraps
    from tenacity import retry, stop_after_attempt, wait_exponential
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {format_exception(e)}")
            raise
    
    return wrapper 