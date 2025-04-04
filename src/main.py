"""
Main entry point for the Multimodal RAG system.
"""
import argparse
import asyncio
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

from loguru import logger

from config.config import (
    CRAWLER_SETTINGS,
    DOCUMENT_PROCESSOR_SETTINGS,
    RETRIEVAL_SETTINGS,
    SCREENSHOTS_DIR,
    PDFS_DIR,
    INDEXES_DIR
)
from src.crawler.crawler import WebsiteCrawler, run_crawler
from src.document_processor.processor import WebpageDocumentProcessor, process_screenshots
from src.retrieval.retriever import VisualRetriever, retrieve_visual_context
from src.response_generator.generator import MultimodalResponseGenerator, generate_response_for_query
from src.utils import ensure_dir

class MultimodalRAG:
    """Main class for the Multimodal RAG system."""
    
    def __init__(self):
        """Initialize the Multimodal RAG system."""
        self.crawler = None
        self.document_processor = None
        self.retriever = None
        self.response_generator = None
        
        # Ensure directories exist
        ensure_dir(SCREENSHOTS_DIR)
        ensure_dir(PDFS_DIR)
        ensure_dir(INDEXES_DIR)
        
        logger.info("Multimodal RAG system initialized")
    
    def _lazy_init_crawler(self) -> WebsiteCrawler:
        """Lazily initialize the crawler.
        
        Returns:
            Initialized crawler
        """
        if self.crawler is None:
            self.crawler = WebsiteCrawler()
        return self.crawler
    
    def _lazy_init_document_processor(self) -> WebpageDocumentProcessor:
        """Lazily initialize the document processor.
        
        Returns:
            Initialized document processor
        """
        if self.document_processor is None:
            self.document_processor = WebpageDocumentProcessor()
        return self.document_processor
    
    def _lazy_init_retriever(self, index_name: Optional[str] = None) -> VisualRetriever:
        """Lazily initialize the retriever.
        
        Args:
            index_name: Optional index name
            
        Returns:
            Initialized retriever
        """
        if self.retriever is None or (index_name and self.retriever.index_name != index_name):
            self.retriever = VisualRetriever(index_name=index_name)
        return self.retriever
    
    def _lazy_init_response_generator(self) -> MultimodalResponseGenerator:
        """Lazily initialize the response generator.
        
        Returns:
            Initialized response generator
        """
        if self.response_generator is None:
            self.response_generator = MultimodalResponseGenerator()
        return self.response_generator
    
    async def crawl_website(self, url: str, max_depth: int = 2, max_pages: int = 20) -> Tuple[List[Path], List[Path]]:
        """Crawl a website and capture screenshots and PDFs.
        
        Args:
            url: URL to crawl
            max_depth: Maximum crawl depth
            max_pages: Maximum number of pages to crawl
            
        Returns:
            Tuple of lists of paths to screenshots and PDFs
        """
        logger.info(f"Crawling website: {url}")
        crawler = self._lazy_init_crawler()
        
        # Crawl the website
        screenshots, pdfs = await crawler.crawl(
            start_urls=[url],
            max_depth=max_depth,
            max_pages=max_pages
        )
        
        logger.info(f"Crawled {len(screenshots)} pages from {url}")
        return screenshots, pdfs
    
    async def process_website_screenshots(self, screenshots: List[Path], index_name: Optional[str] = None) -> str:
        """Process website screenshots and index them.
        
        Args:
            screenshots: List of paths to screenshot images
            index_name: Optional name for the index
            
        Returns:
            Index name/ID
        """
        logger.info(f"Processing {len(screenshots)} website screenshots")
        processor = self._lazy_init_document_processor()
        
        # Use the default index name if not provided
        if index_name is None:
            index_name = RETRIEVAL_SETTINGS["colivara_index_name"]
        
        # Process screenshots
        documents = await processor.process_all(screenshots)
        
        # Index documents
        index_id = await processor.index_documents(documents, index_name)
        
        logger.info(f"Processed and indexed {len(documents)} documents to {index_name}")
        return index_id
    
    async def query(self, query: str, index_name: Optional[str] = None) -> Dict[str, Any]:
        """Query the system with a natural language question.
        
        Args:
            query: Natural language query
            index_name: Optional index name
            
        Returns:
            Dictionary with query, context, and response
        """
        logger.info(f"Processing query: '{query}'")
        start_time = time.time()
        
        # 1. Retrieve visual context
        retriever = self._lazy_init_retriever(index_name)
        results, image_paths = await retriever.retrieve_with_images(query)
        context = retriever.format_retrieval_for_llm(results, image_paths)
        
        retrieval_time = time.time() - start_time
        logger.info(f"Retrieved {len(context['results'])} results in {retrieval_time:.2f}s")
        
        # 2. Generate response
        response_generator = self._lazy_init_response_generator()
        response = await response_generator.generate_response(query, context)
        
        total_time = time.time() - start_time
        logger.info(f"Generated response in {total_time:.2f}s total")
        
        # 3. Return the full result
        result = {
            "query": query,
            "context": context,
            "response": response,
            "metrics": {
                "retrieval_time": retrieval_time,
                "total_time": total_time,
                "num_results": len(context["results"])
            }
        }
        
        return result
    
    async def process_website_and_query(self, url: str, query: str, max_depth: int = 2, 
                                       max_pages: int = 20) -> Dict[str, Any]:
        """Process a website and then query it in one operation.
        
        Args:
            url: Website URL
            query: Natural language query
            max_depth: Maximum crawl depth
            max_pages: Maximum number of pages to crawl
            
        Returns:
            Query result
        """
        logger.info(f"Processing website {url} and querying: '{query}'")
        
        # 1. Crawl the website
        screenshots, _ = await self.crawl_website(url, max_depth, max_pages)
        
        # 2. Process and index screenshots
        index_name = f"website_{int(time.time())}"
        await self.process_website_screenshots(screenshots, index_name)
        
        # 3. Query the indexed website
        result = await self.query(query, index_name)
        
        return result
    
    async def close(self):
        """Close and clean up resources."""
        if self.crawler:
            await self.crawler.close()
        logger.info("Multimodal RAG system closed")

async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Multimodal RAG System for Complex Websites")
    
    # Command subparsers
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Crawl command
    crawl_parser = subparsers.add_parser("crawl", help="Crawl a website")
    crawl_parser.add_argument("url", help="URL to crawl")
    crawl_parser.add_argument("--depth", type=int, default=2, help="Maximum crawl depth")
    crawl_parser.add_argument("--max-pages", type=int, default=20, help="Maximum number of pages to crawl")
    
    # Process command
    process_parser = subparsers.add_parser("process", help="Process screenshots and index them")
    process_parser.add_argument("--screenshots-dir", default=None, help="Directory containing screenshots")
    process_parser.add_argument("--index-name", default=None, help="Name for the index")
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Query the system")
    query_parser.add_argument("query", help="Query to process")
    query_parser.add_argument("--index-name", default=None, help="Name of the index to query")
    
    # Full pipeline command
    pipeline_parser = subparsers.add_parser("pipeline", help="Run the full pipeline")
    pipeline_parser.add_argument("url", help="URL to crawl")
    pipeline_parser.add_argument("query", help="Query to process")
    pipeline_parser.add_argument("--depth", type=int, default=2, help="Maximum crawl depth")
    pipeline_parser.add_argument("--max-pages", type=int, default=20, help="Maximum number of pages to crawl")
    
    # Interactive mode
    interactive_parser = subparsers.add_parser("interactive", help="Run in interactive mode")
    interactive_parser.add_argument("--url", default=None, help="URL to crawl (optional)")
    interactive_parser.add_argument("--index-name", default=None, help="Name of the index to query (optional)")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create the RAG system
    rag = MultimodalRAG()
    
    try:
        if args.command == "crawl":
            screenshots, pdfs = await rag.crawl_website(args.url, args.depth, args.max_pages)
            print(f"Crawled {len(screenshots)} pages")
            print(f"Screenshots saved to: {SCREENSHOTS_DIR}")
            print(f"PDFs saved to: {PDFS_DIR}")
            
        elif args.command == "process":
            # Get screenshots
            if args.screenshots_dir:
                screenshots_dir = Path(args.screenshots_dir)
            else:
                screenshots_dir = Path(SCREENSHOTS_DIR)
            
            screenshots = list(screenshots_dir.glob("*.png"))
            if not screenshots:
                print(f"No screenshots found in {screenshots_dir}")
                return
            
            index_id = await rag.process_website_screenshots(screenshots, args.index_name)
            print(f"Processed {len(screenshots)} screenshots and indexed them with ID: {index_id}")
            
        elif args.command == "query":
            result = await rag.query(args.query, args.index_name)
            print(f"\nQuery: {result['query']}")
            print(f"\nResponse: {result['response']}")
            print(f"\nMetrics: Retrieval time: {result['metrics']['retrieval_time']:.2f}s, "
                  f"Total time: {result['metrics']['total_time']:.2f}s, "
                  f"Results: {result['metrics']['num_results']}")
            
        elif args.command == "pipeline":
            result = await rag.process_website_and_query(args.url, args.query, args.depth, args.max_pages)
            print(f"\nQuery: {result['query']}")
            print(f"\nResponse: {result['response']}")
            print(f"\nMetrics: Retrieval time: {result['metrics']['retrieval_time']:.2f}s, "
                  f"Total time: {result['metrics']['total_time']:.2f}s, "
                  f"Results: {result['metrics']['num_results']}")
            
        elif args.command == "interactive":
            # Interactive mode
            if args.url:
                print(f"Crawling {args.url}...")
                screenshots, _ = await rag.crawl_website(args.url, max_depth=2, max_pages=20)
                index_name = f"website_{int(time.time())}"
                await rag.process_website_screenshots(screenshots, index_name)
                print(f"Crawled and processed {len(screenshots)} pages from {args.url}")
            else:
                index_name = args.index_name
                
            print("\nMultimodal RAG Interactive Mode")
            print("Type 'exit' to quit")
            
            while True:
                query = input("\nEnter your query: ")
                if query.lower() in ["exit", "quit", "q"]:
                    break
                
                result = await rag.query(query, index_name)
                print(f"\nResponse: {result['response']}")
                
        else:
            parser.print_help()
            
    finally:
        # Clean up
        await rag.close()

if __name__ == "__main__":
    asyncio.run(main()) 