"""
Retrieval module for fetching relevant visual contexts based on user queries.
"""
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import colivara
from colivara import Index, Document, SearchResults
from loguru import logger

from config.config import RETRIEVAL_SETTINGS, SCREENSHOTS_DIR
from src.utils import safe_request

class VisualRetriever:
    """Retriever for visual contexts from indexed documents."""
    
    def __init__(self, config: Optional[Dict] = None, index_name: Optional[str] = None):
        """Initialize the retriever.
        
        Args:
            config: Optional configuration to override defaults
            index_name: Optional index name to override default
        """
        self.config = config or RETRIEVAL_SETTINGS
        self.index_name = index_name or self.config["colivara_index_name"]
        
        # Initialize ColiVara client (if not already initialized)
        from config.config import DOCUMENT_PROCESSOR_SETTINGS
        if not hasattr(colivara, "api_key") or not colivara.api_key:
            colivara.api_key = DOCUMENT_PROCESSOR_SETTINGS["colivara_api_key"]
            colivara.base_url = DOCUMENT_PROCESSOR_SETTINGS["colivara_endpoint"]
        
        # Initialize cache if enabled
        self.cache = {} if self.config["use_cache"] else None
        self.cache_ttl = self.config["cache_ttl"]
        
        logger.info(f"Retriever initialized with index: {self.index_name}")
    
    async def _get_index(self) -> Index:
        """Get the index object.
        
        Returns:
            ColiVara Index object
        """
        indexes = await colivara.Index.list()
        index = next((idx for idx in indexes if idx.name == self.index_name), None)
        
        if not index:
            raise ValueError(f"Index '{self.index_name}' not found")
        
        return index
    
    def _check_cache(self, query: str) -> Optional[SearchResults]:
        """Check cache for query results.
        
        Args:
            query: Search query
            
        Returns:
            Cached search results if available and fresh, None otherwise
        """
        if not self.cache:
            return None
        
        cached = self.cache.get(query)
        if not cached:
            return None
        
        # Check if cache is still fresh
        now = time.time()
        if now - cached["timestamp"] > self.cache_ttl:
            # Cache expired
            return None
        
        return cached["results"]
    
    def _update_cache(self, query: str, results: SearchResults) -> None:
        """Update cache with new results.
        
        Args:
            query: Search query
            results: Search results
        """
        if not self.cache:
            return
        
        self.cache[query] = {
            "timestamp": time.time(),
            "results": results
        }
    
    @safe_request
    async def retrieve(self, query: str, limit: int = None, 
                      similarity_threshold: float = None) -> SearchResults:
        """Retrieve relevant documents for a query.
        
        Args:
            query: Search query
            limit: Maximum number of results (overrides config)
            similarity_threshold: Minimum similarity score (overrides config)
            
        Returns:
            Search results with relevant documents
        """
        logger.info(f"Retrieving documents for query: '{query}'")
        
        # Check cache first
        cached_results = self._check_cache(query)
        if cached_results:
            logger.info(f"Returning cached results for query: '{query}'")
            return cached_results
        
        # Get retrieval parameters
        limit = limit or self.config["top_k"]
        similarity_threshold = similarity_threshold or self.config["similarity_threshold"]
        
        # Get the index
        index = await self._get_index()
        
        # Perform search based on retrieval mode
        retrieval_mode = self.config["retrieval_mode"]
        
        if retrieval_mode == "visual":
            # Visual-only search
            results = await index.search(
                query=query,
                limit=limit,
                min_score=similarity_threshold,
                search_type="visual"
            )
        elif retrieval_mode == "text":
            # Text-only search
            results = await index.search(
                query=query,
                limit=limit,
                min_score=similarity_threshold,
                search_type="text"
            )
        elif retrieval_mode == "hybrid":
            # Hybrid search (combines visual and text)
            results = await index.search(
                query=query,
                limit=limit,
                min_score=similarity_threshold,
                search_type="hybrid"
            )
        else:
            raise ValueError(f"Unknown retrieval mode: {retrieval_mode}")
        
        # Rerank results if enabled and we have more than one result
        if self.config["reranking_enabled"] and len(results.documents) > 1:
            reranked_results = await index.rerank(
                query=query,
                documents=results.documents
            )
            results = reranked_results
        
        # Update cache
        self._update_cache(query, results)
        
        logger.info(f"Retrieved {len(results.documents)} documents for query: '{query}'")
        return results
    
    async def retrieve_with_images(self, query: str, limit: int = None,
                                 similarity_threshold: float = None) -> Tuple[SearchResults, Dict[str, Path]]:
        """Retrieve documents with local paths to their source images.
        
        Args:
            query: Search query
            limit: Maximum number of results
            similarity_threshold: Minimum similarity score
            
        Returns:
            Tuple of search results and dictionary mapping document IDs to image paths
        """
        # Get search results
        results = await self.retrieve(query, limit, similarity_threshold)
        
        # Map document IDs to local image paths
        image_paths = {}
        for doc in results.documents:
            # Extract source path from metadata
            source_path = doc.metadata.get("source") if doc.metadata else None
            
            if source_path:
                # Check if the file exists
                path = Path(source_path)
                if path.exists():
                    image_paths[doc.id] = path
                else:
                    # Try to find it in the screenshots directory
                    screenshot_path = Path(SCREENSHOTS_DIR) / path.name
                    if screenshot_path.exists():
                        image_paths[doc.id] = screenshot_path
                    else:
                        logger.warning(f"Could not find image file for document {doc.id}")
        
        logger.info(f"Found {len(image_paths)} image paths for {len(results.documents)} documents")
        return results, image_paths
    
    def format_retrieval_for_llm(self, results: SearchResults, image_paths: Dict[str, Path], 
                               include_images: bool = True) -> Dict[str, Any]:
        """Format retrieval results for consumption by an LLM.
        
        Args:
            results: Search results
            image_paths: Dictionary mapping document IDs to image paths
            include_images: Whether to include images in the context
            
        Returns:
            Formatted context for LLM
        """
        context = {
            "query": results.query,
            "results": []
        }
        
        for i, doc in enumerate(results.documents):
            result = {
                "score": doc.score,
                "metadata": doc.metadata,
                "rank": i + 1,
            }
            
            # Add image path if available
            if doc.id in image_paths and include_images:
                result["image_path"] = str(image_paths[doc.id])
            
            # Add document content
            if hasattr(doc, "content") and doc.content:
                result["content"] = doc.content
            
            # Add document visual elements if available
            if hasattr(doc, "visual_elements") and doc.visual_elements:
                result["visual_elements"] = [
                    {
                        "type": elem.type,
                        "text": elem.text if hasattr(elem, "text") else None,
                        "bounding_box": elem.bounding_box.dict() if hasattr(elem, "bounding_box") else None
                    }
                    for elem in doc.visual_elements
                ]
            
            context["results"].append(result)
        
        return context

async def retrieve_visual_context(query: str, index_name: Optional[str] = None) -> Dict[str, Any]:
    """Retrieve visual context for a query.
    
    Args:
        query: Search query
        index_name: Optional index name
        
    Returns:
        Formatted context for LLM
    """
    retriever = VisualRetriever(index_name=index_name)
    results, image_paths = await retriever.retrieve_with_images(query)
    context = retriever.format_retrieval_for_llm(results, image_paths)
    return context

if __name__ == "__main__":
    # Example usage
    import asyncio
    import json
    
    async def main():
        query = "What are the main navigation options?"
        context = await retrieve_visual_context(query)
        
        # Print results in a readable format
        print(f"Query: {context['query']}")
        print(f"Found {len(context['results'])} results:")
        
        for i, result in enumerate(context["results"]):
            print(f"\nResult {i+1} (score: {result['score']:.2f}):")
            if "image_path" in result:
                print(f"Image: {result['image_path']}")
            if "visual_elements" in result:
                print(f"Visual elements: {len(result['visual_elements'])}")
    
    asyncio.run(main()) 