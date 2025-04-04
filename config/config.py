"""
Configuration settings for the Multimodal RAG system.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
SCREENSHOTS_DIR = DATA_DIR / "screenshots"
PDFS_DIR = DATA_DIR / "pdfs"
INDEXES_DIR = DATA_DIR / "indexes"
LOGS_DIR = BASE_DIR / "logs"

# Create directories if they don't exist
for directory in [SCREENSHOTS_DIR, PDFS_DIR, INDEXES_DIR, LOGS_DIR]:
    directory.mkdir(exist_ok=True, parents=True)

# Crawler settings
CRAWLER_SETTINGS = {
    "headless": True,
    "viewport_width": 1920,
    "viewport_height": 1080,
    "scroll_behavior": "smooth",
    "wait_for_idle": 5,  # seconds
    "max_concurrent_pages": 5,
    "max_pages_per_site": 50,
    "respect_robots_txt": True,
    "user_agent": "MultimodalRAGBot/1.0 (+https://example.com/bot)",
    "proxy": os.getenv("PROXY_URL", None),
    "auth": {
        "username": os.getenv("AUTH_USERNAME", None),
        "password": os.getenv("AUTH_PASSWORD", None),
    },
    "cookies_file": os.getenv("COOKIES_FILE", None),
    "api_key": os.getenv("FIRECRAWL_API_KEY", None)
}

# Document processor settings
DOCUMENT_PROCESSOR_SETTINGS = {
    "image_quality": 90,
    "image_format": "png",
    "max_image_width": 1920,
    "max_image_height": None,  # Maintain aspect ratio
    "pdf_compression": "medium",
    "detect_tables": True,
    "detect_forms": True,
    "detect_images": True,
    "detect_headings": True,
    "colivara_api_key": os.getenv("COLIVARA_API_KEY", ""),
    "colivara_endpoint": os.getenv("COLIVARA_ENDPOINT", "https://api.colivara.ai/v1"),
    "batch_size": 10,
}

# Retrieval settings
RETRIEVAL_SETTINGS = {
    "top_k": 5,
    "similarity_threshold": 0.7,
    "reranking_enabled": True,
    "colivara_index_name": "website_visual_index",
    "cross_modality_enabled": True,
    "retrieval_mode": "visual",  # "visual", "text", or "hybrid"
    "use_cache": True,
    "cache_ttl": 3600,  # seconds
}

# Response generator settings
RESPONSE_GENERATOR_SETTINGS = {
    "model_name": "deepseek-ai/deepseek-janus-pro-1.0",
    "device": "cuda" if os.getenv("USE_GPU", "true").lower() == "true" else "cpu",
    "max_length": 1024,
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 50,
    "repetition_penalty": 1.2,
    "max_batch_size": 1,
    "max_context_length": 4096,
    "trust_remote_code": True,
}

# Logging settings
LOGGING_SETTINGS = {
    "level": os.getenv("LOG_LEVEL", "INFO"),
    "format": "{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    "rotation": "100 MB",
    "retention": "1 month",
} 