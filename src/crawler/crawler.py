"""
Web crawler module for capturing screenshots and generating PDFs of web pages.
"""
import asyncio
import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union
from urllib.parse import urljoin, urlparse

from firecrawl import Crawler, CrawlerConfig
from firecrawl.page_processors import ScreenshotProcessor, PDFProcessor
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
        
        # Configure Firecrawl
        crawler_config = CrawlerConfig(
            headless=self.config["headless"],
            viewport_width=self.config["viewport_width"],
            viewport_height=self.config["viewport_height"],
            user_agent=self.config["user_agent"],
            max_concurrent_pages=self.config["max_concurrent_pages"],
            respect_robots_txt=self.config["respect_robots_txt"],
        )
        
        # Add authentication if provided
        if self.config["auth"]["username"] and self.config["auth"]["password"]:
            crawler_config.set_authentication(
                username=self.config["auth"]["username"],
                password=self.config["auth"]["password"]
            )
        
        # Add proxy if provided
        if self.config["proxy"]:
            crawler_config.set_proxy(self.config["proxy"])
            
        # Create the crawler instance
        self.crawler = Crawler(config=crawler_config)
        
        # Set up processors
        self.screenshot_processor = ScreenshotProcessor(
            output_dir=str(self.screenshots_dir),
            format=DOCUMENT_PROCESSOR_SETTINGS["image_format"],
            quality=DOCUMENT_PROCESSOR_SETTINGS["image_quality"],
            full_page=True
        )
        
        self.pdf_processor = PDFProcessor(
            output_dir=str(self.pdfs_dir)
        )
        
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
                cookies = json.load(f)
                self.crawler.add_cookies(cookies)
                logger.info(f"Loaded {len(cookies)} cookies from {cookies_file}")
        except Exception as e:
            logger.error(f"Failed to load cookies from {cookies_file}: {e}")
    
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
        
        # Set up allowed domains if provided
        domain_filter = None
        if allowed_domains:
            domain_filter = lambda url: any(domain in url for domain in allowed_domains)
        
        # Configure crawler behavior
        self.crawler.configure_crawl(
            start_urls=start_urls,
            max_depth=max_depth,
            max_pages=max_pages,
            url_filter=domain_filter,
            wait_time=self.config["wait_for_idle"],
        )
        
        # Add processors
        self.crawler.add_processor(self.screenshot_processor)
        self.crawler.add_processor(self.pdf_processor)
        
        # Start crawling
        logger.info(f"Starting crawler with {len(start_urls)} seed URLs, max_depth={max_depth}, max_pages={max_pages}")
        result = await self.crawler.run()
        
        # Get the output files
        screenshot_files = [Path(file) for file in self.screenshot_processor.get_output_files()]
        pdf_files = [Path(file) for file in self.pdf_processor.get_output_files()]
        
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
            
            # Capture the page
            page = await self.crawler.browser.new_page()
            
            # Set viewport
            await page.set_viewport_size({
                "width": self.config["viewport_width"], 
                "height": self.config["viewport_height"]
            })
            
            # Navigate to the URL
            await page.goto(url, wait_until="networkidle", timeout=60000)
            
            # Wait for the page to be fully loaded
            await asyncio.sleep(self.config["wait_for_idle"])
            
            # Take screenshot
            await page.screenshot(path=str(screenshot_path), full_page=True)
            
            # Generate PDF
            await page.pdf(path=str(pdf_path))
            
            # Close the page
            await page.close()
            
            logger.info(f"Successfully captured {url}")
            return screenshot_path, pdf_path
            
        except Exception as e:
            logger.error(f"Failed to capture page {url}: {e}")
            return None, None
    
    async def close(self):
        """Close the crawler and release resources."""
        await self.crawler.close()
        logger.info("Crawler closed")

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
        urls = ["https://example.com"]
        screenshot_paths, pdf_paths = await run_crawler(
            start_urls=urls,
            max_depth=1,
            max_pages=5
        )
        print(f"Screenshots: {screenshot_paths}")
        print(f"PDFs: {pdf_paths}")
    
    asyncio.run(main()) 