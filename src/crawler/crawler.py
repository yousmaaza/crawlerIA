"""
Web crawler module using FirecrawlApp with image extraction capabilities.
"""
import asyncio
import json
import os
import base64
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union, Any
from urllib.parse import urlparse, urljoin

from firecrawl import FirecrawlApp
from loguru import logger
import aiohttp
from PIL import Image
import io

from config.config import CRAWLER_SETTINGS, SCREENSHOTS_DIR, PDFS_DIR, IMAGES_DIR
from src.utils import (
    get_clean_filename, ensure_dir, safe_request, 
    image_to_base64, base64_to_image, is_valid_image,
    get_image_dimensions
)

class WebsiteCrawler:
    """Web crawler with image extraction using FirecrawlApp."""

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
        self.images_dir = ensure_dir(IMAGES_DIR)
        
        # API key handling
        api_key = self.config.get("api_key", os.getenv("FIRECRAWL_API_KEY"))
        if not api_key:
            logger.warning("FIRECRAWL_API_KEY environment variable not set. API calls may fail.")
        
        # Create the crawler instance with minimal parameters
        self.crawler = FirecrawlApp(api_key=api_key)
        
        # Create HTTP session for downloading images
        self.session = None
        
        # Load cookies if provided
        if self.config["cookies_file"]:
            self._load_cookies(self.config["cookies_file"])
        
        logger.info("Crawler initialized with image extraction capabilities")

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

    async def _extract_images_from_html(self, html: str, base_url: str) -> List[Dict[str, Any]]:
        """Extract image URLs from HTML content.
        
        Args:
            html: HTML content to extract from
            base_url: Base URL for resolving relative URLs
            
        Returns:
            List of dictionaries with image information
        """
        if not html:
            logger.warning(f"No HTML content to extract images from for {base_url}")
            return []
            
        # Find all image tags using regex
        logger.debug(f"Extracting images from HTML for {base_url}")
        img_pattern = r'<img[^>]+src=["\'](.*?)["\']'
        img_urls = re.findall(img_pattern, html)
        
        # Extract srcset attributes
        srcset_pattern = r'<img[^>]+srcset=["\'](.*?)["\']'
        srcsets = re.findall(srcset_pattern, html)
        
        # Process srcset strings to extract URLs
        for srcset in srcsets:
            # Split the srcset by comma and extract URLs
            for src_item in srcset.split(','):
                url = src_item.strip().split(' ')[0]
                if url:
                    img_urls.append(url)
        
        # Create a set for unique URLs
        unique_img_urls = set()
        
        # Process and resolve URLs
        for url in img_urls:
            # Skip data URLs for now as they're already base64 encoded
            if url.startswith('data:'):
                continue
                
            # Resolve relative URLs
            absolute_url = urljoin(base_url, url)
            unique_img_urls.add(absolute_url)
        
        logger.info(f"Found {len(unique_img_urls)} unique image URLs on {base_url}")
        
        # Check file extensions if needed
        if self.config.get("image_formats"):
            allowed_formats = [f.lower() for f in self.config["image_formats"]]
            filtered_urls = []
            
            for url in unique_img_urls:
                # Extract extension from URL
                path = urlparse(url).path
                ext = os.path.splitext(path)[1].lower().lstrip('.')
                
                if ext in allowed_formats:
                    filtered_urls.append(url)
                    
            logger.debug(f"Filtered to {len(filtered_urls)} images with allowed formats: {allowed_formats}")
            unique_img_urls = filtered_urls
        
        # Return the list of unique image URLs
        return [{"url": url, "source_url": base_url} for url in unique_img_urls]

    async def _download_and_process_image(self, img_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Download and process a single image.
        
        Args:
            img_info: Dictionary with image URL and source information
            
        Returns:
            Dictionary with processed image information or None if failed
        """
        if self.session is None:
            self.session = aiohttp.ClientSession(
                headers={"User-Agent": self.config["user_agent"]}
            )
            
        url = img_info["url"]
        source_url = img_info["source_url"]
        
        try:
            # Download the image
            async with self.session.get(url, timeout=10) as response:
                if response.status != 200:
                    logger.warning(f"Failed to download image from {url}: HTTP {response.status}")
                    return None
                
                # Read image data
                img_data = await response.read()
                
                # Check minimum size requirement
                if len(img_data) < self.config.get("min_image_size", 0):
                    logger.debug(f"Image {url} is too small ({len(img_data)} bytes), skipping")
                    return None
                
                # Create a filename based on URL
                clean_url = get_clean_filename(url)
                image_path = self.images_dir / f"{clean_url}.png"
                
                # Process with PIL to check dimensions and save
                try:
                    img = Image.open(io.BytesIO(img_data))
                    
                    # Check dimensions
                    min_width = self.config.get("min_image_width", 0)
                    min_height = self.config.get("min_image_height", 0)
                    
                    if img.width < min_width or img.height < min_height:
                        logger.debug(f"Image {url} dimensions ({img.width}x{img.height}) below minimum requirements, skipping")
                        return None
                    
                    # Save the image
                    img.save(image_path)
                    logger.debug(f"Saved image to {image_path}")
                    
                    # Create result dictionary
                    result = {
                        "url": url,
                        "source_url": source_url,
                        "path": str(image_path),
                        "width": img.width,
                        "height": img.height,
                        "format": img.format,
                        "size_bytes": len(img_data)
                    }
                    
                    return result
                    
                except Exception as e:
                    logger.warning(f"Failed to process image from {url}: {e}")
                    return None
                    
        except Exception as e:
            logger.warning(f"Error downloading image from {url}: {e}")
            return None

    async def _process_images(self, images_info: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process and download multiple images concurrently.
        
        Args:
            images_info: List of dictionaries with image information
            
        Returns:
            List of processed image information
        """
        # Limit the number of images to process
        max_images = self.config.get("max_images_per_page", 20)
        if len(images_info) > max_images:
            logger.info(f"Limiting to {max_images} images out of {len(images_info)} found")
            images_info = images_info[:max_images]
        
        # Process images concurrently
        tasks = []
        for img_info in images_info:
            tasks.append(self._download_and_process_image(img_info))
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out failures and exceptions
        processed_images = []
        for result in results:
            if isinstance(result, Exception):
                logger.warning(f"Error processing image: {result}")
            elif result is not None:
                processed_images.append(result)
        
        logger.info(f"Successfully processed {len(processed_images)} out of {len(images_info)} images")
        return processed_images

    async def _capture_screenshot(self, url: str) -> Optional[Path]:
        """Capture a screenshot of a webpage using Firecrawl.
        
        Args:
            url: URL to capture
            
        Returns:
            Path to saved screenshot or None if failed
        """
        try:
            # Generate filename
            filename = get_clean_filename(url)
            screenshot_path = self.screenshots_dir / f"{filename}.png"
            
            # Create screenshot scraper parameters
            # This is a separate API call specifically for screenshots
            # Adjust as needed based on Firecrawl's actual API
            try:
                # Try to use a dedicated screenshot method if available
                if hasattr(self.crawler, 'capture_screenshot') and callable(getattr(self.crawler, 'capture_screenshot')):
                    screenshot_data = self.crawler.capture_screenshot(url)
                else:
                    # Fall back to standard scrape but try to extract screenshot if included
                    logger.debug("No dedicated screenshot method found, using standard scrape")
                    result = self.crawler.scrape_url(url=url)
                    screenshot_data = result.get("screenshot") if isinstance(result, dict) else None
                    
                if not screenshot_data:
                    logger.warning(f"No screenshot data returned for {url}")
                    return None
            except Exception as e:
                logger.error(f"Failed to capture screenshot for {url}: {e}")
                return None
                
            # Save the screenshot
            if isinstance(screenshot_data, str) and screenshot_data.startswith("data:image"):
                # Convert base64 to image and save
                img = base64_to_image(screenshot_data)
                img.save(screenshot_path)
            elif isinstance(screenshot_data, bytes):
                # Save binary data directly
                with open(screenshot_path, "wb") as f:
                    f.write(screenshot_data)
            else:
                logger.warning(f"Unsupported screenshot format: {type(screenshot_data)}")
                return None
                
            logger.info(f"Saved screenshot to {screenshot_path}")
            return screenshot_path
                
        except Exception as e:
            logger.error(f"Failed to capture screenshot for {url}: {e}")
            return None

    async def crawl_url(self, url: str) -> Dict:
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
        
        # Call the FirecrawlApp API
        try:
            # Just use standard scrape_url without the screenshot parameter
            result = self.crawler.scrape_url(url=url)
            
            # Log the type of content received
            logger.debug(f"Result type: {type(result)}")
            
            # Save the full result as JSON
            json_path = self.output_dir / f"{filename}.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2)
            logger.info(f"Saved full result to {json_path}")
            
            # Initialize saved files list
            saved_files = [json_path]
            
            # Capture screenshot as a separate step
            screenshot_path = await self._capture_screenshot(url)
            if screenshot_path:
                saved_files.append(screenshot_path)
            
            # Extract markdown content if available
            if isinstance(result, dict) and 'markdown' in result and result['markdown']:
                markdown_path = self.output_dir / f"{filename}.md"
                with open(markdown_path, 'w', encoding='utf-8') as f:
                    f.write(result['markdown'])
                logger.info(f"Saved markdown content to {markdown_path}")
                saved_files.append(markdown_path)
            
            # Extract HTML content and images if available
            extracted_images = []
            if isinstance(result, dict) and 'html' in result and result['html']:
                # Save HTML
                html_path = self.output_dir / f"{filename}.html"
                with open(html_path, 'w', encoding='utf-8') as f:
                    f.write(result['html'])
                logger.info(f"Saved HTML content to {html_path}")
                saved_files.append(html_path)
                
                # Extract images if enabled
                if self.config.get("extract_images", True):
                    logger.info(f"Extracting images from {url}")
                    image_infos = await self._extract_images_from_html(result['html'], url)
                    if image_infos:
                        extracted_images = await self._process_images(image_infos)
                        logger.info(f"Extracted {len(extracted_images)} images from {url}")
            
            return {
                'url': url,
                'result': result,
                'saved_files': saved_files,
                'screenshot': screenshot_path,
                'extracted_images': extracted_images
            }
            
        except Exception as e:
            logger.error(f"Error crawling {url}: {e}")
            return {
                'url': url,
                'error': str(e),
                'saved_files': [],
                'screenshot': None,
                'extracted_images': []
            }

    @safe_request
    async def crawl(self,
                   start_urls: List[str],
                   max_depth: int = 2,
                   max_pages: Optional[int] = None,
                   allowed_domains: Optional[List[str]] = None) -> Tuple[List[Path], List[Path], List[Dict]]:
        """Crawl websites, capture screenshots and extract images.

        Args:
            start_urls: List of URLs to start crawling from
            max_depth: Maximum crawl depth
            max_pages: Maximum number of pages to crawl (overrides config)
            allowed_domains: List of domains to restrict crawling to

        Returns:
            Tuple containing lists of paths to saved files, screenshots, and image information
        """
        max_pages = max_pages or self.config["max_pages_per_site"]

        # Start crawling
        logger.info(f"Starting crawler with {len(start_urls)} seed URLs, max_depth={max_depth}, max_pages={max_pages}")

        # Initialize empty lists for results
        all_saved_files = []
        all_screenshots = []
        all_images = []
        
        for url in start_urls:
            try:
                # Process each URL
                result = await self.crawl_url(url)
                
                # Collect results
                all_saved_files.extend(result.get('saved_files', []))
                
                if result.get('screenshot'):
                    all_screenshots.append(result['screenshot'])
                    
                all_images.extend(result.get('extracted_images', []))
                
                # Wait briefly between requests
                await asyncio.sleep(1)
                    
            except Exception as e:
                logger.error(f"Error processing URL {url}: {e}")

        # Log summary
        logger.info(f"Crawling complete. Saved {len(all_saved_files)} files")
        logger.info(f"Captured {len(all_screenshots)} screenshots")
        logger.info(f"Extracted {len(all_images)} images")

        # Return the paths to the generated files and images
        return all_saved_files, all_screenshots, all_images

    async def capture_single_page(self, url: str) -> Tuple[Optional[Path], Optional[Path], List[Dict]]:
        """Capture a single page without crawling links.

        Args:
            url: URL of the page to capture

        Returns:
            Tuple of paths to saved files, screenshot, and extracted images
        """
        logger.info(f"Capturing single page: {url}")

        try:
            # Process the URL
            result = await self.crawl_url(url)
            
            # Extract the relevant information
            saved_files = result.get('saved_files', [])
            screenshot = result.get('screenshot')
            extracted_images = result.get('extracted_images', [])
            
            # Return the first saved file, screenshot, and extracted images
            primary_file = saved_files[0] if saved_files else None
            
            return primary_file, screenshot, extracted_images

        except Exception as e:
            logger.error(f"Failed to capture page {url}: {e}")
            return None, None, []

    async def close(self):
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
                
        # Close HTTP session if it exists
        if self.session is not None:
            try:
                await self.session.close()
                logger.info("HTTP session closed")
            except Exception as e:
                logger.error(f"Error closing HTTP session: {e}")

async def run_crawler(start_urls: List[str], **kwargs) -> Tuple[List[Path], List[Path], List[Dict]]:
    """Run the crawler as a standalone function.

    Args:
        start_urls: List of URLs to crawl
        **kwargs: Additional parameters for the crawler

    Returns:
        Tuple containing lists of paths to saved files, screenshots, and image information
    """
    crawler = WebsiteCrawler()
    try:
        saved_files, screenshots, extracted_images = await crawler.crawl(start_urls, **kwargs)
        return saved_files, screenshots, extracted_images
    finally:
        await crawler.close()

if __name__ == "__main__":
    # Example usage
    import asyncio
    
    async def main():
        # URLs to crawl
        urls = ["https://example.com", "https://docs.python.org/3/"]
        
        saved_files, screenshots, extracted_images = await run_crawler(
            start_urls=urls,
            max_depth=1,
            max_pages=5
        )
        logger.info(f"Saved files: {saved_files}")
        logger.info(f"Screenshots: {screenshots}")
        logger.info(f"Extracted {len(extracted_images)} images")

    asyncio.run(main())
