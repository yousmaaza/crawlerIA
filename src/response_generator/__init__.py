"""
Response generator package for the multimodal RAG system.

This module handles generating responses for user queries based on retrieved visual contexts
using the DeepSeek-Janus Pro multimodal LLM.
"""
from src.response_generator.generator import MultimodalResponseGenerator, generate_response_for_query

__all__ = ["MultimodalResponseGenerator", "generate_response_for_query"]
