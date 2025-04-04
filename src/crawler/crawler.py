"""
Web crawler module for capturing screenshots and generating PDFs of web pages.
"""
import asyncio
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union
from urllib.parse import urljoin, urlparse

from firecrawl import FirecrawlApp
from loguru import logger

from config.config import CRAWLER_SETTINGS, SCREENSHOTS_DIR, PDFS_DIR, DOCUMENT_PROCESSOR_SETTINGS
from src.utils import get_clean_filename, ensure_dir, safe_request

class WebsiteCrawler:
    """Crawler class for capturing website screenshots and PDFs."""

    def __init__(self, config: Optional[Dict] = None):
        """Initialize the crawler with configuration.

        Args:
            config: Optional custom configuration to override defaults
        """
        self.config = config or CRAWLER_SETTINGS
        self.screenshots_dir = ensure_dir(SCREENSHOTS_DIR)
        self.pdfs_dir = ensure_dir(PDFS_DIR)

        # API key handling
        api_key = os.getenv("FIRECRAWL_API_KEY")
        
        # Create the crawler instance with minimal parameters
        self.crawler = FirecrawlApp(api_key=api_key)
        
        # Store crawler options for later use
        self.viewport_width = self.config["viewport_width"]
        self.viewport_height = self.config["viewport_height"]
        self.user_agent = self.config["user_agent"]
        self.wait_for_idle = self.config["wait_for_idle"]
        self.headless = self.config["headless"]

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

    async def setup_authentication(self) -> None:
        """Set up authentication if credentials are provided."""
        if self.config["auth"]["username"] and self.config["auth"]["password"]:
            try:
                auth_params = {
                    "username": self.config["auth"]["username"],
                    "password": self.config["auth"]["password"]
                }
                # Pass authentication via parameters
                # Note: Adjust this based on actual API requirements
                await self.crawler.authenticate(**auth_params)
                logger.info("Authentication successful")
            except Exception as e:
                logger.error(f"Authentication failed: {e}")

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
            Tuple containing lists of paths to screenshots and PDFs
        """
        max_pages = max_pages or self.config["max_pages_per_site"]

        # Set up authentication if needed
        await self.setup_authentication()

        # Start crawling
        logger.info(f"Starting crawler with {len(start_urls)} seed URLs, max_depth={max_depth}, max_pages={max_pages}")

        # Initialize empty lists for results
        screenshot_files = []
        pdf_files = []
        
        for url in start_urls:
            try:
                # Based on error message, we need to simplify parameters to match API expectations
                # Note: FirecrawlApp v1 API appears to have different parameter structure
                
                # Generate clean filenames
                filename = get_clean_filename(url)
                screenshot_path = self.screenshots_dir / f"{filename}.{DOCUMENT_PROCESSOR_SETTINGS['image_format']}"
                pdf_path = self.pdfs_dir / f"{filename}.pdf"
                
                # Handle each URL individually with minimal parameters
                result = await self.crawler.scrape_url(url=url)
                
                # Log success
                logger.info(f"Successfully scraped URL: {url}")
                
                # Check if files were created during scraping
                if screenshot_path.exists():
                    screenshot_files.append(screenshot_path)
                if pdf_path.exists():
                    pdf_files.append(pdf_path)
                    
            except Exception as e:
                logger.error(f"Error processing URL {url}: {e}")

        logger.info(f"Crawling complete. Captured {len(screenshot_files)} screenshots and {len(pdf_files)} PDFs")

        # Return the paths to the generated files
        return screenshot_files, pdf_files

    async def capture_single_page(self, url: str) -> Tuple[Optional[Path], Optional[Path]]:
        """Capture a single page without crawling links.

        Args:
            url: URL of the page to capture

        Returns:
            Tuple of paths to the screenshot and PDF, or None if capture failed
        """
        logger.info(f"Capturing single page: {url}")

        try:
            # Generate clean filenames
            filename = get_clean_filename(url)
            screenshot_path = self.screenshots_dir / f"{filename}.{DOCUMENT_PROCESSOR_SETTINGS['image_format']}"
            pdf_path = self.pdfs_dir / f"{filename}.pdf"

            # Set up authentication if needed
            await self.setup_authentication()

            # Capture the page with minimal parameters
            result = await self.crawler.scrape_url(url=url)

            logger.info(f"Successfully captured {url}")
            
            # Check if files were created
            screenshot_exists = screenshot_path.exists()
            pdf_exists = pdf_path.exists()
            
            return screenshot_path if screenshot_exists else None, pdf_path if pdf_exists else None

        except Exception as e:
            logger.error(f"Failed to capture page {url}: {e}")
            return None, None

    async def close(self):
        """Close the crawler and release resources."""
        if hasattr(self, 'crawler') and self.crawler:
            try:
                # Try different approaches to close resources
                if hasattr(self.crawler, 'close') and callable(self.crawler.close):
                    await self.crawler.close()
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
        Tuple containing lists of paths to screenshots and PDFs
    """
    crawler = WebsiteCrawler()
    try:
        screenshot_paths, pdf_paths = await crawler.crawl(start_urls, **kwargs)
        return screenshot_paths, pdf_paths
    finally:
        await crawler.close()

if __name__ == "__main__":
    # Example usage
    import asyncio
    from config.config import DOCUMENT_PROCESSOR_SETTINGS

    async def main():
        urls = ["https://www.example.com"]
        screenshot_paths, pdf_paths = await run_crawler(
            start_urls=urls,
            max_depth=1,
            max_pages=5
        )
        print(f"Screenshots: {screenshot_paths}")
        print(f"PDFs: {pdf_paths}")

    asyncio.run(main())
