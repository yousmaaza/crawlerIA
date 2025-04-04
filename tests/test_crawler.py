"""
Tests for the web crawler module.
"""
import os
import asyncio
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from loguru import logger

from src.crawler.crawler import WebsiteCrawler, run_crawler
from config.config import CRAWLER_SETTINGS, SCREENSHOTS_DIR


class TestWebsiteCrawler(unittest.TestCase):
    """Test cases for the WebsiteCrawler class."""

    def setUp(self):
        """Set up test environment before each test."""
        # Disable logger for testing
        logger.remove()
        logger.add(lambda _: None, level="ERROR")
        
        # Create a mock config for testing
        self.test_config = {
            "headless": True,
            "viewport_width": 800,
            "viewport_height": 600,
            "scroll_behavior": "smooth",
            "wait_for_idle": 2,
            "max_concurrent_pages": 2,
            "max_pages_per_site": 5,
            "respect_robots_txt": True,
            "user_agent": "TestBot/1.0",
            "proxy": None,
            "auth": {
                "username": None,
                "password": None,
            },
            "cookies_file": None,
            "api_key": "test_api_key",
            "output_dir": Path("./test_output")
        }
        
        # Create test directories if they don't exist
        self.test_output_dir = Path("./test_output")
        self.test_output_dir.mkdir(exist_ok=True)
        
        # Set up test URL
        self.test_url = "https://example.com"

    def tearDown(self):
        """Clean up after each test."""
        # Remove test output directory
        if self.test_output_dir.exists():
            for file in self.test_output_dir.glob("*"):
                file.unlink()
            self.test_output_dir.rmdir()

    @patch('src.crawler.crawler.FirecrawlApp')
    def test_crawler_initialization(self, mock_firecrawl):
        """Test crawler initialization with custom config."""
        # Setup mock
        mock_firecrawl.return_value = MagicMock()
        
        # Create crawler with test config
        crawler = WebsiteCrawler(config=self.test_config)
        
        # Assert FirecrawlApp was instantiated with the correct API key
        mock_firecrawl.assert_called_once_with(api_key=self.test_config["api_key"])
        
        # Check that output directories were created
        self.assertTrue(crawler.output_dir.exists())
        self.assertTrue(Path(SCREENSHOTS_DIR).exists())
        
        # Check that config was set correctly
        self.assertEqual(crawler.config, self.test_config)

    @patch('src.crawler.crawler.FirecrawlApp')
    def test_authentication_setup(self, mock_firecrawl):
        """Test authentication setup."""
        # Create a mock instance
        mock_instance = MagicMock()
        mock_firecrawl.return_value = mock_instance
        
        # Setup test config with auth credentials
        auth_config = self.test_config.copy()
        auth_config["auth"]["username"] = "testuser"
        auth_config["auth"]["password"] = "testpass"
        
        # Create crawler with auth config
        crawler = WebsiteCrawler(config=auth_config)
        
        # Call authentication method
        crawler.setup_authentication()
        
        # Assert authenticate was called with correct credentials
        mock_instance.authenticate.assert_called_once_with(
            username="testuser", password="testpass"
        )

    @patch('src.crawler.crawler.WebsiteCrawler.crawl_url')
    @pytest.mark.asyncio
    async def test_crawl_method(self, mock_crawl_url):
        """Test the crawl method."""
        # Setup mock for crawl_url
        mock_crawl_url.return_value = {
            'url': self.test_url,
            'saved_files': [Path("./test_output/example.com_home.json")],
            'result': {'html': '<html></html>', 'markdown': '# Example'}
        }
        
        # Create crawler with test config
        crawler = WebsiteCrawler(config=self.test_config)
        
        # Call the crawl method
        saved_files, _ = await crawler.crawl(
            start_urls=[self.test_url],
            max_depth=1,
            max_pages=2
        )
        
        # Assert crawl_url was called with the correct URL
        mock_crawl_url.assert_called_once_with(self.test_url)
        
        # Check results
        self.assertEqual(len(saved_files), 1)
        self.assertEqual(saved_files[0], Path("./test_output/example.com_home.json"))

    @patch('src.crawler.crawler.WebsiteCrawler')
    @pytest.mark.asyncio
    async def test_run_crawler_function(self, mock_crawler_class):
        """Test the run_crawler standalone function."""
        # Setup mocks
        mock_instance = MagicMock()
        mock_crawler_class.return_value = mock_instance
        
        mock_instance.crawl.return_value = (
            [Path("./test_output/example.com_home.json")], []
        )
        
        # Call standalone function
        saved_files, _ = await run_crawler(
            start_urls=[self.test_url],
            max_depth=1,
            max_pages=2
        )
        
        # Assert WebsiteCrawler was instantiated
        mock_crawler_class.assert_called_once()
        
        # Assert crawl method was called with correct parameters
        mock_instance.crawl.assert_called_once_with(
            [self.test_url],
            max_depth=1,
            max_pages=2
        )
        
        # Check that close was called
        mock_instance.close.assert_called_once()
        
        # Check results
        self.assertEqual(len(saved_files), 1)
        self.assertEqual(saved_files[0], Path("./test_output/example.com_home.json"))


if __name__ == '__main__':
    pytest.main(['-xvs', 'tests/test_crawler.py'])