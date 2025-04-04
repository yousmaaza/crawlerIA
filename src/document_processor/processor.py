"""
Document processor module for processing webpage screenshots using ColiVara.
"""
import asyncio
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import colivara
from colivara import DocumentProcessor, ProcessingConfig, Document
from loguru import logger
from PIL import Image

from config.config import DOCUMENT_PROCESSOR_SETTINGS, SCREENSHOTS_DIR
from src.utils import batch_items, safe_request

class WebpageDocumentProcessor:
    """Document processor for webpage screenshots using ColiVara."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the document processor.
        
        Args:
            config: Optional configuration to override defaults
        """
        self.config = config or DOCUMENT_PROCESSOR_SETTINGS
        
        # Initialize ColiVara client
        colivara.api_key = self.config["colivara_api_key"]
        colivara.base_url = self.config["colivara_endpoint"]
        
        # Create processing configuration
        self.processing_config = ProcessingConfig(
            detect_tables=self.config["detect_tables"],
            detect_forms=self.config["detect_forms"],
            detect_images=self.config["detect_images"],
            detect_headings=self.config["detect_headings"],
        )
        
        # Initialize document processor
        self.processor = DocumentProcessor(config=self.processing_config)
        
        logger.info("Document processor initialized")
    
    @safe_request
    async def process_image(self, image_path: Union[str, Path]) -> Document:
        """Process a single image using ColiVara.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Processed document
        """
        logger.info(f"Processing image: {image_path}")
        image_path = Path(image_path)
        
        # Open and prepare the image
        with Image.open(image_path) as img:
            # Resize if needed
            if self.config["max_image_width"] and img.width > self.config["max_image_width"]:
                ratio = self.config["max_image_width"] / img.width
                new_height = int(img.height * ratio)
                img = img.resize((self.config["max_image_width"], new_height))
                
            # Process the image
            document = await self.processor.process_image(
                image=img,
                document_id=image_path.stem,
                metadata={"source": str(image_path), "type": "webpage"}
            )
            
            logger.info(f"Successfully processed {image_path}")
            return document
    
    async def process_batch(self, image_paths: List[Union[str, Path]]) -> List[Document]:
        """Process a batch of images.
        
        Args:
            image_paths: List of paths to images
            
        Returns:
            List of processed documents
        """
        logger.info(f"Processing batch of {len(image_paths)} images")
        
        tasks = [self.process_image(image_path) for image_path in image_paths]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions
        documents = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Failed to process {image_paths[i]}: {result}")
            else:
                documents.append(result)
        
        logger.info(f"Successfully processed {len(documents)} out of {len(image_paths)} images")
        return documents
    
    async def process_all(self, image_paths: List[Union[str, Path]]) -> List[Document]:
        """Process all images in batches.
        
        Args:
            image_paths: List of paths to images
            
        Returns:
            List of all processed documents
        """
        logger.info(f"Processing {len(image_paths)} images in batches of {self.config['batch_size']}")
        
        all_documents = []
        batches = batch_items(image_paths, self.config["batch_size"])
        
        for i, batch in enumerate(batches):
            logger.info(f"Processing batch {i+1}/{len(batches)}")
            batch_documents = await self.process_batch(batch)
            all_documents.extend(batch_documents)
            
            # Add a small delay between batches to avoid rate limiting
            if i < len(batches) - 1:
                await asyncio.sleep(1)
        
        logger.info(f"Completed processing {len(all_documents)} documents")
        return all_documents

    async def index_documents(self, documents: List[Document], index_name: str) -> str:
        """Index processed documents in ColiVara.
        
        Args:
            documents: List of documents to index
            index_name: Name of the index
            
        Returns:
            Index ID
        """
        logger.info(f"Indexing {len(documents)} documents to index '{index_name}'")
        
        # Create or get existing index
        try:
            index = await colivara.Index.create(name=index_name)
            logger.info(f"Created new index: {index_name}")
        except colivara.errors.ConflictError:
            # Index already exists
            indexes = await colivara.Index.list()
            index = next((idx for idx in indexes if idx.name == index_name), None)
            logger.info(f"Using existing index: {index_name}")
        
        if not index:
            raise ValueError(f"Failed to create or retrieve index '{index_name}'")
        
        # Add documents to index
        for i, document in enumerate(documents):
            try:
                await index.add_document(document)
                logger.debug(f"Indexed document {i+1}/{len(documents)}: {document.id}")
            except Exception as e:
                logger.error(f"Failed to index document {document.id}: {e}")
        
        logger.info(f"Completed indexing {len(documents)} documents")
        return index.id

async def process_screenshots(screenshot_paths: List[Path], index_name: Optional[str] = None) -> Tuple[List[Document], str]:
    """Process screenshots and index them.
    
    Args:
        screenshot_paths: List of paths to screenshot images
        index_name: Optional name for the index (default is from config)
        
    Returns:
        Tuple containing list of processed documents and index ID
    """
    from config.config import RETRIEVAL_SETTINGS
    
    processor = WebpageDocumentProcessor()
    
    # Process all screenshots
    documents = await processor.process_all(screenshot_paths)
    
    # Index the documents
    index_name = index_name or RETRIEVAL_SETTINGS["colivara_index_name"]
    index_id = await processor.index_documents(documents, index_name)
    
    return documents, index_id

if __name__ == "__main__":
    # Example usage
    import asyncio
    from pathlib import Path
    
    async def main():
        # Example with some sample screenshots
        screenshot_dir = Path(SCREENSHOTS_DIR)
        screenshot_paths = list(screenshot_dir.glob("*.png"))
        
        if not screenshot_paths:
            print("No screenshots found. Please run the crawler first.")
            return
        
        print(f"Found {len(screenshot_paths)} screenshots")
        documents, index_id = await process_screenshots(screenshot_paths)
        print(f"Processed {len(documents)} documents and indexed them with ID: {index_id}")
    
    asyncio.run(main()) 