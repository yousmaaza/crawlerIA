"""
Web crawler module for capturing screenshots and generating PDFs of web pages.
"""
import asyncio
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union
from urllib.parse import urljoin, urlparse

from firecrawl import FirecrawlClient, CaptureOptions, CrawlOptions, CookieFormat
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
        crawler_options = {
            "headless": self.config["headless"],
            "viewport": {
                "width": self.config["viewport_width"],
                "height": self.config["viewport_height"]
            },
            "user_agent": self.config["user_agent"],
            "max_concurrent_pages": self.config["max_concurrent_pages"],
            "respect_robots_txt": self.config["respect_robots_txt"],
        }

        # Add proxy if provided
        if self.config["proxy"]:
            crawler_options["proxy"] = {
                "server": self.config["proxy"]
            }

        # Create the crawler instance
        self.crawler = FirecrawlClient(**crawler_options)

        # Configure capture options
        self.capture_options = CaptureOptions(
            screenshots=True,
            pdfs=True,
            wait_for_network_idle=True,
            wait_time=self.config["wait_for_idle"],
            screenshot_format=DOCUMENT_PROCESSOR_SETTINGS["image_format"],
            screenshot_quality=DOCUMENT_PROCESSOR_SETTINGS["image_quality"],
            full_page=True,
            output_dir=str(self.screenshots_dir.parent),  # Parent directory that contains both screenshots and pdfs
            screenshot_dir=self.screenshots_dir.name,
            pdf_dir=self.pdfs_dir.name
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
                cookies_data = json.load(f)
                # Determine the format based on the structure
                if isinstance(cookies_data, list) and all("name" in cookie and "value" in cookie for cookie in cookies_data):
                    cookies_format = CookieFormat.NETSCAPE
                else:
                    cookies_format = CookieFormat.JSON

                self.crawler.load_cookies(cookies_data, format=cookies_format)
                logger.info(f"Loaded cookies from {cookies_file}")
        except Exception as e:
            logger.error(f"Failed to load cookies from {cookies_file}: {e}")

    async def setup_authentication(self) -> None:
        """Set up authentication if credentials are provided."""
        if self.config["auth"]["username"] and self.config["auth"]["password"]:
            try:
                await self.crawler.login(
                    username=self.config["auth"]["username"],
                    password=self.config["auth"]["password"]
                )
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

        # Configure crawl options
        crawl_options = CrawlOptions(
            max_depth=max_depth,
            max_pages=max_pages,
            allowed_domains=allowed_domains,
            respect_robots_txt=self.config["respect_robots_txt"],
            wait_time=self.config["wait_for_idle"]
        )

        # Start crawling
        logger.info(f"Starting crawler with {len(start_urls)} seed URLs, max_depth={max_depth}, max_pages={max_pages}")

        # Run the crawler
        result = await self.crawler.crawl(
            urls=start_urls,
            options=crawl_options,
            capture_options=self.capture_options
        )

        # Extract file paths from results
        screenshot_files = []
        pdf_files = []
        
        # Process results to get file paths
        for page_result in result.pages:
            if page_result.screenshot:
                screenshot_files.append(Path(page_result.screenshot))
            if page_result.pdf:
                pdf_files.append(Path(page_result.pdf))

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

            # Capture the page
            result = await self.crawler.capture(
                url=url,
                options=self.capture_options,
                output_screenshot=str(screenshot_path),
                output_pdf=str(pdf_path)
            )

            logger.info(f"Successfully captured {url}")
            return screenshot_path if result.screenshot else None, pdf_path if result.pdf else None

        except Exception as e:
            logger.error(f"Failed to capture page {url}: {e}")
            return None, None

    async def close(self):
        """Close the crawler and release resources."""
        if hasattr(self, 'crawler') and self.crawler:
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
        urls = ["https://www.example.com"]
        screenshot_paths, pdf_paths = await run_crawler(
            start_urls=urls,
            max_depth=1,
            max_pages=5
        )
        print(f"Screenshots: {screenshot_paths}")
        print(f"PDFs: {pdf_paths}")

    asyncio.run(main())
