"""
Web crawler module for capturing screenshots and generating PDFs of web pages.
"""
import asyncio
import json
import os
import time
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
        if not api_key:
            logger.warning("FIRECRAWL_API_KEY environment variable not set. API calls may fail.")
        
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
                # Call synchronous method - FirecrawlApp seems to use synchronous methods
                self.crawler.authenticate(**auth_params)
                logger.info("Authentication successful")
            except Exception as e:
                logger.error(f"Authentication failed: {e}")

    def scrape_with_retry(self, url: str, screenshot_path: str, pdf_path: str, max_retries: int = 3, retry_delay: int = 2) -> Dict:
        """Attempt to scrape a URL with retries on failure.
        
        Args:
            url: URL to scrape
            screenshot_path: Path where to save the screenshot
            pdf_path: Path where to save the PDF
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
            
        Returns:
            Result dictionary or None if all attempts fail
        """
        logger.info(f"Scraping URL: {url}")
        logger.info(f"Screenshot will be saved to: {screenshot_path}")
        logger.info(f"PDF will be saved to: {pdf_path}")
        
        attempts = 0
        while attempts < max_retries:
            try:
                # Try with minimal parameters but explicitly specify output paths
                result = self.crawler.scrape_url(
                    url=url,
                    output_screenshot=screenshot_path,
                    output_pdf=pdf_path
                )
                logger.info(f"Scrape successful, result type: {type(result)}")
                return result
            except Exception as e:
                attempts += 1
                logger.warning(f"Attempt {attempts}/{max_retries} failed for {url}: {e}")
                if attempts < max_retries:
                    # Exponential backoff
                    wait_time = retry_delay * (2 ** (attempts - 1))
                    logger.info(f"Retrying in {wait_time} seconds...")
                    # Use time.sleep for synchronous delay
                    time.sleep(wait_time)
                else:
                    logger.error(f"All {max_retries} attempts failed for {url}")
                    raise

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
                # Generate clean filenames
                filename = get_clean_filename(url)
                screenshot_path = self.screenshots_dir / f"{filename}.{DOCUMENT_PROCESSOR_SETTINGS['image_format']}"
                pdf_path = self.pdfs_dir / f"{filename}.pdf"
                
                # Convert paths to strings for FirecrawlApp
                screenshot_path_str = str(screenshot_path)
                pdf_path_str = str(pdf_path)
                
                # Try to scrape with retries - synchronous call, no await
                result = self.scrape_with_retry(
                    url, 
                    screenshot_path=screenshot_path_str, 
                    pdf_path=pdf_path_str
                )
                
                # Wait a moment for files to be written - use asyncio.sleep since we're in an async method
                await asyncio.sleep(3)
                
                # Check if files were created during scraping
                if screenshot_path.exists():
                    logger.info(f"Screenshot saved: {screenshot_path}")
                    screenshot_files.append(screenshot_path)
                else:
                    logger.warning(f"Screenshot not found at expected path: {screenshot_path}")
                    logger.info("Checking filesystem for any similar files...")
                    
                    # Look for any PNG files in the screenshots directory with a similar name
                    for file in self.screenshots_dir.glob(f"*{filename}*.png"):
                        logger.info(f"Found similar screenshot: {file}")
                        screenshot_files.append(file)
                        break
                
                if pdf_path.exists():
                    logger.info(f"PDF saved: {pdf_path}")
                    pdf_files.append(pdf_path)
                else:
                    logger.warning(f"PDF not found at expected path: {pdf_path}")
                    logger.info("Checking filesystem for any similar files...")
                    
                    # Look for any PDF files in the pdfs directory with a similar name
                    for file in self.pdfs_dir.glob(f"*{filename}*.pdf"):
                        logger.info(f"Found similar PDF: {file}")
                        pdf_files.append(file)
                        break
                    
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

            # Convert paths to strings for FirecrawlApp
            screenshot_path_str = str(screenshot_path)
            pdf_path_str = str(pdf_path)

            # Set up authentication if needed
            await self.setup_authentication()

            # Try to scrape with retries - synchronous call with explicit output paths
            result = self.scrape_with_retry(
                url, 
                screenshot_path=screenshot_path_str, 
                pdf_path=pdf_path_str
            )
            
            # Wait a moment for files to be written
            await asyncio.sleep(3)
            
            # Check if files were created
            screenshot_exists = screenshot_path.exists()
            pdf_exists = pdf_path.exists()
            
            # Look for similar files if exact matches not found
            if not screenshot_exists:
                for file in self.screenshots_dir.glob(f"*{filename}*.png"):
                    screenshot_path = file
                    screenshot_exists = True
                    break
            
            if not pdf_exists:
                for file in self.pdfs_dir.glob(f"*{filename}*.pdf"):
                    pdf_path = file
                    pdf_exists = True
                    break
            
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
                    # Call close synchronously if it exists
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
        # Use a simpler example site that's likely to work
        urls = ["https://example.com"]
        screenshot_paths, pdf_paths = await run_crawler(
            start_urls=urls,
            max_depth=1,
            max_pages=5
        )
        print(f"Screenshots: {screenshot_paths}")
        print(f"PDFs: {pdf_paths}")

    asyncio.run(main())
