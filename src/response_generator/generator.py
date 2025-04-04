"""
Response generator module using DeepSeek-Janus Pro to generate responses based on visual contexts.
"""
import base64
import io
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import torch
from loguru import logger
from PIL import Image
from transformers import (
    AutoProcessor, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig
)

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
        self.processor = None
        
        # Lazy initialization - will load when needed
        logger.info(f"Response generator initialized (model will be loaded on first use)")
    
    def _initialize_model(self) -> None:
        """Initialize the DeepSeek-Janus model and processor."""
        try:
            logger.info(f"Loading model: {self.config['model_name']}")
            
            # Configure quantization if using GPU
            if "cuda" in self.config["device"] and torch.cuda.is_available():
                # Use 4-bit quantization for memory efficiency
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True
                )
            else:
                quantization_config = None
            
            # Load processor (handles both text tokenization and image processing)
            self.processor = AutoProcessor.from_pretrained(
                self.config["model_name"],
                trust_remote_code=self.config["trust_remote_code"]
            )
            
            # Load model with quantization if available
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config["model_name"],
                device_map=self.config["device"],
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                quantization_config=quantization_config,
                trust_remote_code=self.config["trust_remote_code"]
            )
            
            logger.info(f"Model loaded successfully on {self.config['device']}")
            
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            raise
    
    def _load_image(self, image_path: Union[str, Path]) -> Image.Image:
        """Load an image from path.
        
        Args:
            image_path: Path to the image
            
        Returns:
            Loaded PIL Image
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        try:
            # Open the image
            img = Image.open(image_path)
            # Convert to RGB if needed (DeepSeek-Janus expects RGB)
            if img.mode != "RGB":
                img = img.convert("RGB")
            return img
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            raise
    
    def _prepare_inputs(self, query: str, image_paths: List[Union[str, Path]]) -> Dict[str, Any]:
        """Prepare inputs for the model.
        
        Args:
            query: User query
            image_paths: List of paths to images
            
        Returns:
            Prepared inputs for the model
        """
        # Load images
        images = []
        for path in image_paths[:4]:  # Limit to 4 images (model constraint)
            try:
                img = self._load_image(path)
                images.append(img)
            except Exception as e:
                logger.warning(f"Skipping image {path}: {e}")
        
        # Check if we have images
        if not images:
            # Text-only mode
            inputs = self.processor(
                text=query, 
                return_tensors="pt"
            )
            return inputs
        
        # Prepare multimodal inputs
        prompt = f"Look at the images and answer this question: {query}"
        inputs = self.processor(
            text=prompt,
            images=images,
            return_tensors="pt"
        )
        
        # Move inputs to the same device as the model
        if self.model is not None and hasattr(self.model, "device"):
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        return inputs
    
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
        if self.model is None or self.processor is None:
            self._initialize_model()
        
        # Extract image paths from context
        image_paths = []
        for result in context.get("results", []):
            if "image_path" in result:
                image_paths.append(result["image_path"])
        
        try:
            # Prepare inputs
            inputs = self._prepare_inputs(query, image_paths)
            
            # Generate response
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config["max_length"],
                    temperature=self.config["temperature"],
                    top_p=self.config["top_p"],
                    top_k=self.config["top_k"],
                    repetition_penalty=self.config["repetition_penalty"],
                    do_sample=True
                )
            
            # Process output
            if "input_ids" in inputs:
                # Skip the input tokens to get only the generated response
                response_ids = outputs[0][len(inputs["input_ids"][0]):]
            else:
                response_ids = outputs[0]
            
            # Decode the response
            response = self.processor.decode(
                response_ids,
                skip_special_tokens=True
            )
            
            logger.info(f"Response generated successfully with {len(response)} characters")
            return response
            
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            return f"I apologize, but I encountered an error while processing your request: {str(e)}"

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
        print(f"\\nQuery: {query}")
        print(f"\\nResponse: {response}")
    
    asyncio.run(main())
