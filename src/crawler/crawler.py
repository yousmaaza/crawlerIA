"""
Simplified web crawler module using FirecrawlApp.
"""
import asyncio
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union
from urllib.parse import urlparse

from firecrawl import FirecrawlApp
from loguru import logger

from config.config import CRAWLER_SETTINGS, SCREENSHOTS_DIR, PDFS_DIR
from src.utils import get_clean_filename, ensure_dir, safe_request

class WebsiteCrawler:
    """Simple crawler to extract website content using FirecrawlApp."""

    def __init__(self, config: Optional[Dict] = None):
        """Initialize the crawler with configuration.

        Args:
            config: Optional custom configuration to override defaults
        """
        self.config = config or CRAWLER_SETTINGS
        
        # Set up output directories
        self.output_dir = ensure_dir(Path(self.config.get("output_dir", "./output")))
        self.screenshots_dir = ensure_dir(SCREENSHOTS_DIR)
        self.pdfs_dir = ensure_dir(PDFS_DIR)
        
        # API key handling
        api_key = os.getenv("FIRECRAWL_API_KEY")
        if not api_key:
            logger.warning("FIRECRAWL_API_KEY environment variable not set. API calls may fail.")
        
        # Create the crawler instance with minimal parameters
        self.crawler = FirecrawlApp(api_key=api_key)
        
        # Load cookies if provided
        if self.config["cookies_file"]:
            self._load_cookies(self.config["cookies_file"])

    def _load_cookies(self, cookies_file: str) -> None:
        """Load cookies from file.

        Args:
            cookies_file: Path to cookies file
        """
        try:
            with open(cookies_file, 'r') as f:
                cookies_data = json.load(f)
                self.crawler.add_cookies(cookies_data)
                logger.info(f"Loaded cookies from {cookies_file}")
        except Exception as e:
            logger.error(f"Failed to load cookies from {cookies_file}: {e}")

    def setup_authentication(self) -> None:
        """Set up authentication if credentials are provided."""
        if self.config["auth"]["username"] and self.config["auth"]["password"]:
            try:
                auth_params = {
                    "username": self.config["auth"]["username"],
                    "password": self.config["auth"]["password"]
                }
                # Call synchronous method
                self.crawler.authenticate(**auth_params)
                logger.info("Authentication successful")
            except Exception as e:
                logger.error(f"Authentication failed: {e}")

    def crawl_url(self, url: str) -> Dict:
        """Crawl a single URL and extract content.
        
        Args:
            url: The URL to crawl
            
        Returns:
            Dictionary with extracted content and saved file paths
        """
        logger.info(f"Crawling URL: {url}")
        
        # Set up authentication if needed
        self.setup_authentication()
        
        # Generate clean filename from URL
        filename = get_clean_filename(url)
        
        # Call the FirecrawlApp API with minimal parameters
        try:
            # Just pass the URL - no additional parameters
            result = self.crawler.scrape_url(url=url)
            
            # Log the type of content received
            logger.info(f"Result type: {type(result)}")
            if isinstance(result, dict):
                logger.info(f"Result contains keys: {list(result.keys())}")
            
            # Save the full result as JSON
            json_path = self.output_dir / f"{filename}.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2)
            logger.info(f"Saved full result to {json_path}")
            
            # Extract markdown content if available
            saved_files = [json_path]
            if isinstance(result, dict) and 'markdown' in result and result['markdown']:
                markdown_path = self.output_dir / f"{filename}.md"
                with open(markdown_path, 'w', encoding='utf-8') as f:
                    f.write(result['markdown'])
                logger.info(f"Saved markdown content to {markdown_path}")
                saved_files.append(markdown_path)
            
            # Extract HTML content if available
            if isinstance(result, dict) and 'html' in result and result['html']:
                html_path = self.output_dir / f"{filename}.html"
                with open(html_path, 'w', encoding='utf-8') as f:
                    f.write(result['html'])
                logger.info(f"Saved HTML content to {html_path}")
                saved_files.append(html_path)
            
            return {
                'url': url,
                'result': result,
                'saved_files': saved_files
            }
            
        except Exception as e:
            logger.error(f"Error crawling {url}: {e}")
            return {
                'url': url,
                'error': str(e),
                'saved_files': []
            }

    @safe_request
    async def crawl(self,
                   start_urls: List[str],
                   max_depth: int = 2,
                   max_pages: Optional[int] = None,
                   allowed_domains: Optional[List[str]] = None) -> Tuple[List[Path], List[Path]]:
        """Crawl websites, capture screenshots and generate PDFs.

        Args:
            start_urls: List of URLs to start crawling from
            max_depth: Maximum crawl depth
            max_pages: Maximum number of pages to crawl (overrides config)
            allowed_domains: List of domains to restrict crawling to

        Returns:
            Tuple containing lists of paths to saved files
        """
        max_pages = max_pages or self.config["max_pages_per_site"]

        # Start crawling
        logger.info(f"Starting crawler with {len(start_urls)} seed URLs, max_depth={max_depth}, max_pages={max_pages}")

        # Initialize empty lists for results
        all_saved_files = []
        
        for url in start_urls:
            try:
                # Process each URL
                result = self.crawl_url(url)
                all_saved_files.extend(result.get('saved_files', []))
                
                # Wait briefly between requests
                await asyncio.sleep(1)
                    
            except Exception as e:
                logger.error(f"Error processing URL {url}: {e}")

        logger.info(f"Crawling complete. Saved {len(all_saved_files)} files")

        # Return the paths to the generated files
        return all_saved_files, []  # Return empty list as second item for backward compatibility

    async def capture_single_page(self, url: str) -> Tuple[Optional[Path], Optional[Path]]:
        """Capture a single page without crawling links.

        Args:
            url: URL of the page to capture

        Returns:
            Tuple of paths to the saved files, or None if capture failed
        """
        logger.info(f"Capturing single page: {url}")

        try:
            # Process the URL
            result = self.crawl_url(url)
            saved_files = result.get('saved_files', [])
            
            # Return the first saved file as the primary result, or None
            primary_file = saved_files[0] if saved_files else None
            secondary_file = saved_files[1] if len(saved_files) > 1 else None
            
            return primary_file, secondary_file

        except Exception as e:
            logger.error(f"Failed to capture page {url}: {e}")
            return None, None

    def close(self):
        """Close the crawler and release resources."""
        if hasattr(self, 'crawler') and self.crawler:
            try:
                # Try to close resources
                if hasattr(self.crawler, 'close') and callable(self.crawler.close):
                    self.crawler.close()
                    logger.info("Crawler closed")
                else:
                    logger.info("Crawler does not have a close method, resources may not be properly released")
            except Exception as e:
                logger.error(f"Error closing crawler: {e}")

async def run_crawler(start_urls: List[str], **kwargs) -> Tuple[List[Path], List[Path]]:
    """Run the crawler as a standalone function.

    Args:
        start_urls: List of URLs to crawl
        **kwargs: Additional parameters for the crawler

    Returns:
        Tuple containing lists of paths to saved files
    """
    crawler = WebsiteCrawler()
    try:
        saved_files, _ = await crawler.crawl(start_urls, **kwargs)
        return saved_files, []  # Return empty list as second item for backward compatibility
    finally:
        crawler.close()

if __name__ == "__main__":
    # Example usage
    import asyncio
    
    async def main():
        # URLs to crawl
        urls = ["https://example.com", "https://docs.python.org/3/"]
        
        saved_files, _ = await run_crawler(
            start_urls=urls,
            max_depth=1,
            max_pages=5
        )
        print(f"Saved files: {saved_files}")

    asyncio.run(main())
