"""
Retrieval package for the multimodal RAG system.

This module handles retrieving relevant visual contexts based on user queries
using ColiVara's visual understanding capabilities.
"""
from src.retrieval.retriever import VisualRetriever, retrieve_visual_context

__all__ = ["VisualRetriever", "retrieve_visual_context"]
