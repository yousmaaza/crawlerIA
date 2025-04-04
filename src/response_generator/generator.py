"""
Response generator module using DeepSeek-Janus Pro to generate responses based on visual contexts.
"""
import base64
import io
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import torch
from loguru import logger
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from config.config import RESPONSE_GENERATOR_SETTINGS

class MultimodalResponseGenerator:
    """Response generator using DeepSeek-Janus Pro for visual context understanding."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the response generator.
        
        Args:
            config: Optional configuration to override defaults
        """
        self.config = config or RESPONSE_GENERATOR_SETTINGS
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        
        # Initialize model and tokenizer
        self._initialize_model()
        
        logger.info(f"Response generator initialized with model: {self.config['model_name']}")
    
    def _initialize_model(self) -> None:
        """Initialize the DeepSeek-Janus model and tokenizer."""
        try:
            logger.info(f"Loading model: {self.config['model_name']}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config['model_name'],
                trust_remote_code=self.config['trust_remote_code']
            )
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config['model_name'],
                torch_dtype=torch.bfloat16 if "cuda" in self.config['device'] else torch.float32,
                device_map=self.config['device'],
                trust_remote_code=self.config['trust_remote_code']
            )
            
            logger.info(f"Model loaded successfully on {self.config['device']}")
            
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            raise
    
    def _encode_image(self, image_path: Union[str, Path]) -> str:
        """Encode image to base64 for model input.
        
        Args:
            image_path: Path to the image
            
        Returns:
            Base64 encoded image
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        with open(image_path, "rb") as img_file:
            encoded_img = base64.b64encode(img_file.read()).decode('utf-8')
        
        return encoded_img
    
    def _prepare_input_with_visual_context(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare input with visual context for the model.
        
        Args:
            query: User query
            context: Visual context from retriever
            
        Returns:
            Prepared input for the model
        """
        # Extract image paths from context
        image_paths = []
        for result in context.get("results", []):
            if "image_path" in result:
                image_paths.append(result["image_path"])
        
        # If no images found, run in text-only mode
        if not image_paths:
            logger.warning("No images found in context, running in text-only mode")
            return {
                "text": f"USER: {query}\nASSISTANT:"
            }
        
        # Use at most 4 images (model constraint)
        image_paths = image_paths[:4]
        
        # Encode images
        encoded_images = []
        for path in image_paths:
            try:
                encoded_img = self._encode_image(path)
                encoded_images.append(encoded_img)
            except Exception as e:
                logger.error(f"Failed to encode image {path}: {e}")
        
        # If no images could be encoded, fall back to text-only mode
        if not encoded_images:
            logger.warning("Failed to encode any images, falling back to text-only mode")
            return {
                "text": f"USER: {query}\nASSISTANT:"
            }
        
        # Prepare the input
        input_data = {
            "text": f"USER: {query}\nASSISTANT:",
            "images": encoded_images
        }
        
        return input_data
    
    def _prepare_prompt_with_visual_elements(self, query: str, context: Dict[str, Any]) -> str:
        """Prepare a prompt that includes descriptions of visual elements.
        
        Args:
            query: User query
            context: Visual context from retriever
            
        Returns:
            Prompt string with visual element descriptions
        """
        prompt_parts = [
            f"I'll describe the visual content I'm looking at to answer your question: \"{query}\""
        ]
        
        # Add descriptions of visual elements in the context
        for i, result in enumerate(context.get("results", [])):
            visual_elements = result.get("visual_elements", [])
            if visual_elements:
                prompt_parts.append(f"\nIn result {i+1}, I can see:")
                
                # Group elements by type
                elements_by_type = {}
                for elem in visual_elements:
                    elem_type = elem.get("type", "unknown")
                    if elem_type not in elements_by_type:
                        elements_by_type[elem_type] = []
                    elements_by_type[elem_type].append(elem)
                
                # Add descriptions for each type
                for elem_type, elements in elements_by_type.items():
                    prompt_parts.append(f"- {len(elements)} {elem_type}s")
                    # Add details for certain element types
                    if elem_type in ["heading", "paragraph", "link", "button"]:
                        texts = [e.get("text") for e in elements if e.get("text")]
                        if texts:
                            prompt_parts.append(f"  Text content includes: {', '.join(texts[:5])}")
                            if len(texts) > 5:
                                prompt_parts.append(f"  ...and {len(texts) - 5} more")
        
        # Add the actual query
        prompt_parts.append(f"\nNow, to answer your question about: {query}")
        
        return "\n".join(prompt_parts)
    
    async def generate_response(self, query: str, context: Dict[str, Any]) -> str:
        """Generate a response using DeepSeek-Janus based on the query and visual context.
        
        Args:
            query: User query
            context: Visual context from the retriever
            
        Returns:
            Generated response text
        """
        logger.info(f"Generating response for query: '{query}'")
        
        # Check if model is loaded
        if self.model is None or self.tokenizer is None:
            self._initialize_model()
        
        # Prepare input with visual context
        input_data = self._prepare_input_with_visual_context(query, context)
        
        try:
            # Generate response
            if "images" in input_data:
                # Multimodal mode
                logger.info(f"Using multimodal mode with {len(input_data['images'])} images")
                
                # Format the input for DeepSeek-Janus
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": query}
                        ]
                    }
                ]
                
                # Add images to the message
                for image_b64 in input_data["images"]:
                    messages[0]["content"].append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_b64}"}
                    })
                
                # Generate response
                response = self.model.chat(
                    self.tokenizer,
                    messages,
                    temperature=self.config["temperature"],
                    top_p=self.config["top_p"],
                    max_new_tokens=self.config["max_length"]
                )
                
                generated_text = response.get("content", "")
                
            else:
                # Text-only mode
                logger.info("Using text-only mode")
                
                # Use text-only generation
                messages = [
                    {"role": "user", "content": query}
                ]
                
                response = self.model.chat(
                    self.tokenizer,
                    messages,
                    temperature=self.config["temperature"],
                    top_p=self.config["top_p"],
                    max_new_tokens=self.config["max_length"]
                )
                
                generated_text = response.get("content", "")
            
            logger.info(f"Response generated successfully with {len(generated_text)} characters")
            return generated_text
            
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            return f"I apologize, but I encountered an error while processing your request. Please try again or rephrase your question."

async def generate_response_for_query(query: str, context: Dict[str, Any]) -> str:
    """Generate a response for a query with visual context.
    
    Args:
        query: User query
        context: Visual context from retriever
        
    Returns:
        Generated response text
    """
    generator = MultimodalResponseGenerator()
    response = await generator.generate_response(query, context)
    return response

if __name__ == "__main__":
    # Example usage
    import asyncio
    from src.retrieval.retriever import retrieve_visual_context
    
    async def main():
        query = "What are the main sections of this webpage?"
        
        # Get visual context
        context = await retrieve_visual_context(query)
        
        # Generate response
        response = await generate_response_for_query(query, context)
        print(f"\nQuery: {query}")
        print(f"\nResponse: {response}")
    
    asyncio.run(main()) 