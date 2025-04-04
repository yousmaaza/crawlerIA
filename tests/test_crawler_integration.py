"""
Integration tests for the web crawler module.
"""
import asyncio
import os
from pathlib import Path

import pytest
from loguru import logger

from src.crawler.crawler import WebsiteCrawler
from config.config import CRAWLER_SETTINGS, SCREENSHOTS_DIR, PDFS_DIR

# Skip these tests if the FIRECRAWL_API_KEY is not set
pytestmark = pytest.mark.skipif(
    os.getenv("FIRECRAWL_API_KEY") is None,
    reason="FIRECRAWL_API_KEY environment variable not set"
)


class TestCrawlerIntegration:
    """Integration tests for the WebsiteCrawler."""

    @classmethod
    def setup_class(cls):
        """Set up test environment once before all tests."""
        # Disable logger for testing
        logger.remove()
        logger.add(lambda _: None, level="INFO")
        
        # Create test directories if they don't exist
        cls.screenshots_dir = Path(SCREENSHOTS_DIR)
        cls.pdfs_dir = Path(PDFS_DIR)
        cls.screenshots_dir.mkdir(exist_ok=True)
        cls.pdfs_dir.mkdir(exist_ok=True)
        
        # Make note of starting files to clean up only new ones later
        cls.original_screenshots = set(cls.screenshots_dir.glob("*"))
        cls.original_pdfs = set(cls.pdfs_dir.glob("*"))
        
        # Set up test URLs
        cls.test_url = "https://example.com"  # Simple static site for basic testing
        cls.simple_dynamic_url = "https://quotes.toscrape.com/"  # Simple dynamic site

    @classmethod
    def teardown_class(cls):
        """Clean up after all tests."""
        # Remove only files created during tests
        current_screenshots = set(cls.screenshots_dir.glob("*"))
        current_pdfs = set(cls.pdfs_dir.glob("*"))
        
        # Delete new screenshots
        for file in current_screenshots - cls.original_screenshots:
            if file.is_file():
                file.unlink()
        
        # Delete new PDFs
        for file in current_pdfs - cls.original_pdfs:
            if file.is_file():
                file.unlink()

    @pytest.mark.asyncio
    async def test_crawl_static_site(self):
        """Test crawling a simple static website."""
        crawler = WebsiteCrawler()
        
        try:
            # Crawl example.com (1 page)
            saved_files, _ = await crawler.crawl(
                start_urls=[self.test_url],
                max_depth=1,
                max_pages=1
            )
            
            # Check that we got some files
            assert len(saved_files) > 0, "No files were saved from crawling example.com"
            
            # Check file existence
            for file_path in saved_files:
                assert file_path.exists(), f"Saved file {file_path} does not exist"
            
            # Verify content of at least one file
            if saved_files:
                first_file = saved_files[0]
                with open(first_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    assert len(content) > 0, "Saved file is empty"
                    
                    # If it's a JSON file, it should at least contain the URL
                    if first_file.suffix == '.json':
                        assert self.test_url in content, f"URL {self.test_url} not found in saved content"
        
        finally:
            # Ensure crawler is closed
            crawler.close()

    @pytest.mark.asyncio
    async def test_capture_single_page(self):
        """Test capturing a single page."""
        crawler = WebsiteCrawler()
        
        try:
            # Capture a single page
            primary_file, secondary_file = await crawler.capture_single_page(self.test_url)
            
            # Check that we got at least one file
            assert primary_file is not None, "No primary file was saved"
            assert primary_file.exists(), f"Primary file {primary_file} does not exist"
            
            # Check content of the primary file
            with open(primary_file, 'r', encoding='utf-8') as f:
                content = f.read()
                assert len(content) > 0, "Primary file is empty"
        
        finally:
            # Ensure crawler is closed
            crawler.close()

    @pytest.mark.asyncio
    async def test_crawl_with_depth(self):
        """Test crawling with depth > 1 on a site with multiple pages."""
        # Skip if running in CI environment to avoid long-running tests
        if os.getenv("CI") == "true":
            pytest.skip("Skipping depth crawl test in CI environment")
        
        crawler = WebsiteCrawler()
        
        try:
            # Crawl quotes.toscrape.com with depth 2 (should get multiple pages)
            saved_files, _ = await crawler.crawl(
                start_urls=[self.simple_dynamic_url],
                max_depth=2,
                max_pages=5  # Limit to 5 pages to keep test short
            )
            
            # Check that we got multiple files (should be more than just the start page)
            assert len(saved_files) > 1, "Not enough files were saved from depth crawling"
            
            # Check file existence
            for file_path in saved_files:
                assert file_path.exists(), f"Saved file {file_path} does not exist"
        
        finally:
            # Ensure crawler is closed
            crawler.close()


if __name__ == '__main__':
    pytest.main(['-xvs', 'tests/test_crawler_integration.py'])