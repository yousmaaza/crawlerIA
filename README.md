# Multimodal RAG System for Complex Websites

A Python-based Retrieval-Augmented Generation (RAG) system that effectively processes and understands complex website layouts without traditional text scraping and chunking. This system preserves visual context and layout information for proper understanding of web content.

## Features

- **Vision-based RAG**: Uses visual understanding instead of traditional text chunking
- **Preserves Layout and Context**: Maintains relationships between elements on the page
- **No OCR Required**: Works directly with visual content
- **Handles Complex Layouts**: Properly interprets tables, images, and complex UI elements
- **Human-like Understanding**: Processes content similar to how humans perceive it

## Architecture

The system consists of four main components:

1. **Web Crawler** (using Firecrawl)
   - Captures high-quality screenshots of webpages
   - Generates PDFs for archiving
   - Handles pagination, dynamic content, and authentication

2. **Document Processor** (using ColiVara)
   - Indexes webpage screenshots as images
   - Preserves visual layout and content relationships
   - Understands document structure without OCR

3. **Retrieval System** (using ColiVara)
   - Fetches relevant visual contexts based on user queries
   - Provides similarity-based ranking
   - Supports both visual and hybrid search modes

4. **Response Generator** (using DeepSeek-Janus Pro)
   - Generates responses based on retrieved visual contexts
   - Leverages multimodal understanding
   - Runs locally via Hugging Face Transformers

## Requirements

- Python 3.9+
- See `requirements.txt` for all dependencies

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/multimodal-rag.git
   cd multimodal-rag
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   Create a `.env` file in the project root with the following:
   ```
   COLIVARA_API_KEY=your_api_key_here
   COLIVARA_ENDPOINT=https://api.colivara.ai/v1
   USE_GPU=true  # Set to false for CPU-only mode
   LOG_LEVEL=INFO
   ```

## Usage

The system can be used in multiple ways:

### Command Line Interface

1. **Crawl a website**:
   ```
   python src/main.py crawl https://example.com --depth 2 --max-pages 20
   ```

2. **Process screenshots**:
   ```
   python src/main.py process --screenshots-dir data/screenshots --index-name my_website
   ```

3. **Query the indexed website**:
   ```
   python src/main.py query "What is the main product offered?" --index-name my_website
   ```

4. **Full pipeline (crawl, process, and query)**:
   ```
   python src/main.py pipeline https://example.com "What are the main navigation options?"
   ```

5. **Interactive mode**:
   ```
   python src/main.py interactive --url https://example.com
   ```

### Python API

You can also use the system programmatically:

```python
import asyncio
from src.main import MultimodalRAG

async def example():
    rag = MultimodalRAG()
    
    # Crawl a website
    screenshots, pdfs = await rag.crawl_website("https://example.com", max_depth=2, max_pages=20)
    
    # Process screenshots
    index_id = await rag.process_website_screenshots(screenshots, index_name="example_com")
    
    # Query the system
    result = await rag.query("What is the main product offered?", index_name="example_com")
    
    print(result["response"])
    
    # Clean up
    await rag.close()

# Run the example
asyncio.run(example())
```

## Handling Common Challenges

### Authentication

For websites requiring authentication:

1. Set credentials in your `.env` file:
   ```
   AUTH_USERNAME=your_username
   AUTH_PASSWORD=your_password
   ```

2. Or provide a cookies file:
   ```
   COOKIES_FILE=path/to/cookies.json
   ```

### Dynamic Content

The system handles dynamic content by:

- Waiting for network idle state
- Scrolling to reveal lazy-loaded elements
- Supporting JavaScript execution
- Configurable wait times before capture

### Pagination

For websites with pagination:

1. Increase the crawl depth:
   ```
   python src/main.py crawl https://example.com --depth 3
   ```

2. Customize the crawler to follow pagination links:
   ```python
   # Example custom pagination handling in your code
   crawler.configure_crawl(
       start_urls=[url],
       url_filter=lambda url: "page=" in url or "pagination" in url
   )
   ```

## Project Structure

```
.
├── config/             # Configuration settings
├── data/               # Data storage
│   ├── indexes/        # ColiVara indexes
│   ├── pdfs/           # Generated PDFs
│   └── screenshots/    # Website screenshots
├── logs/               # Log files
├── src/                # Source code
│   ├── crawler/        # Website crawler module
│   ├── document_processor/ # Document processing module
│   ├── response_generator/ # Response generation module
│   ├── retrieval/      # Retrieval system module
│   ├── utils.py        # Utility functions
│   └── main.py         # Main entry point
├── tests/              # Test files
├── .env                # Environment variables
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Firecrawl](https://github.com/example/firecrawl) for web crawling
- [ColiVara](https://colivara.ai) for document understanding and retrieval
- [DeepSeek](https://github.com/deepseek-ai/DeepSeek-Janus) for the Janus Pro multimodal LLM
- [Hugging Face Transformers](https://huggingface.co/docs/transformers) for model integration 