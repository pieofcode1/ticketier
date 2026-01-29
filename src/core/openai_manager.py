"""
OpenAI Manager for Azure OpenAI Responses API

This module provides a unified interface for interacting with Azure OpenAI
using the Responses API for both text and image-based completions.
"""

import os
import base64
from typing import List, Dict, Any, Optional, Union, Literal
from dataclasses import dataclass
from enum import Enum
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider


class ContentType(Enum):
    """Enum for content types in messages."""
    TEXT = "text"
    IMAGE_URL = "image_url"
    IMAGE_FILE = "image_file"


@dataclass
class ImageContent:
    """Represents image content for vision models."""
    data: bytes
    media_type: str = "image/png"
    detail: Literal["low", "high", "auto"] = "auto"
    
    def to_base64(self) -> str:
        """Convert image bytes to base64 string."""
        return base64.b64encode(self.data).decode("utf-8")
    
    def to_data_url(self) -> str:
        """Convert image to data URL format."""
        return f"data:{self.media_type};base64,{self.to_base64()}"


@dataclass
class ResponseResult:
    """Standardized result from OpenAI responses."""
    content: str
    response_id: str
    model: str
    usage: Dict[str, int]
    raw_response: Any


class OpenAIManager:
    """
    Manager class for Azure OpenAI Responses API operations.
    
    This class provides a unified interface for:
    - Text completions using the Responses API
    - Vision/Image analysis using multimodal models
    - Embedding generation
    - Response retrieval and management
    
    Example:
        ```python
        manager = OpenAIManager()
        
        # Simple text completion
        result = manager.complete("What is the capital of France?")
        
        # With system instructions
        result = manager.complete(
            "Tell me a joke",
            instructions="You are a comedian who tells dad jokes"
        )
        
        # Image analysis
        with open("image.png", "rb") as f:
            image_data = f.read()
        result = manager.analyze_image(image_data, "Describe this image")
        ```
    """

    def __init__(
        self,
        azure_openai_endpoint: Optional[str] = None,
        azure_openai_key: Optional[str] = None,
        api_version: Optional[str] = None,
        gpt_deployment: Optional[str] = None,
        embedding_deployment: Optional[str] = None,
        use_azure_credential: Optional[bool] = None,
    ):
        """
        Initialize the OpenAI Manager.
        
        All parameters can be provided directly or read from environment variables.

        Args:
            azure_openai_endpoint: Azure OpenAI endpoint (env: AI_FOUNDRY_OPENAI_ENDPOINT)
            azure_openai_key: Azure OpenAI API key (env: AI_FOUNDRY_API_KEY)
            api_version: API version for Responses API (default: "preview")
            gpt_deployment: GPT model deployment name (env: GPT_MODEL_DEPLOYMENT_NAME)
            embedding_deployment: Embedding model deployment (env: EMBEDDING_DEPLOYMENT_NAME)
            use_azure_credential: Use DefaultAzureCredential (env: USE_AZURE_CREDENTIAL)
        """
        # Load configuration from environment or parameters
        self.azure_openai_endpoint = azure_openai_endpoint or os.getenv("AI_FOUNDRY_OPENAI_ENDPOINT")
        azure_openai_key = azure_openai_key or os.getenv("AI_FOUNDRY_API_KEY")
        # Responses API requires "preview" as the api_version, not the standard dated versions
        self.api_version = api_version or os.getenv("RESPONSES_API_VERSION", "preview")
        self.gpt_deployment = gpt_deployment or os.getenv("GPT_MODEL_DEPLOYMENT_NAME", "gpt-4.1")
        self.embedding_deployment = embedding_deployment or os.getenv("EMBEDDING_DEPLOYMENT_NAME", "text-embedding-ada-002")
        use_azure_credential = use_azure_credential if use_azure_credential is not None else os.getenv("USE_AZURE_CREDENTIAL", "false").lower() == "true"
        
        # Validate required parameters
        if not self.azure_openai_endpoint:
            raise ValueError(
                "azure_openai_endpoint must be provided or set AI_FOUNDRY_OPENAI_ENDPOINT environment variable"
            )
        
        # Ensure endpoint has proper format for Responses API
        base_url = self.azure_openai_endpoint.rstrip("/")
        if not base_url.endswith("openai/v1"):
            base_url = f"{base_url}/openai/v1/"
        else:
            base_url = f"{base_url}/"
            
        # Initialize OpenAI client
        if use_azure_credential:
            token_provider = get_bearer_token_provider(
                DefaultAzureCredential(),
                "https://cognitiveservices.azure.com/.default"
            )
            self.client = AzureOpenAI(
                base_url=base_url,
                azure_ad_token_provider=token_provider,
                api_version=self.api_version
            )
        else:
            if not azure_openai_key:
                raise ValueError(
                    "azure_openai_key must be provided or set AI_FOUNDRY_API_KEY environment variable "
                    "when not using Azure credential"
                )
            self.client = AzureOpenAI(
                base_url=base_url,
                api_key=azure_openai_key,
                api_version=self.api_version
            )

    def complete(
        self,
        prompt: str,
        instructions: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> ResponseResult:
        """
        Generate a text completion using the Responses API.
        
        Args:
            prompt: The user's input/question
            instructions: System instructions for the model
            model: Model deployment name (defaults to gpt_deployment)
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens in the response
            **kwargs: Additional parameters passed to the API
            
        Returns:
            ResponseResult with the generated content and metadata
        """
        model = model or self.gpt_deployment
        
        # Build request parameters
        request_params = {
            "model": model,
            "input": prompt,
            **kwargs
        }
        
        if instructions:
            request_params["instructions"] = instructions
        if temperature is not None:
            request_params["temperature"] = temperature
        if max_tokens is not None:
            request_params["max_output_tokens"] = max_tokens
            
        response = self.client.responses.create(**request_params)
        
        return self._parse_response(response)

    def complete_with_messages(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> ResponseResult:
        """
        Generate a completion using chat message format.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
                Example: [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "Hello!"}
                ]
            model: Model deployment name
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            **kwargs: Additional parameters
            
        Returns:
            ResponseResult with the generated content
        """
        model = model or self.gpt_deployment
        
        request_params = {
            "model": model,
            "input": messages,
            **kwargs
        }
        
        if temperature is not None:
            request_params["temperature"] = temperature
        if max_tokens is not None:
            request_params["max_output_tokens"] = max_tokens
            
        response = self.client.responses.create(**request_params)
        
        return self._parse_response(response)

    def analyze_image(
        self,
        image: Union[bytes, str, ImageContent],
        prompt: str,
        instructions: Optional[str] = None,
        model: Optional[str] = None,
        detail: Literal["low", "high", "auto"] = "auto",
        media_type: str = "image/png",
        **kwargs
    ) -> ResponseResult:
        """
        Analyze an image using a vision-capable model.
        
        Args:
            image: Image data as bytes, base64 string, URL, or ImageContent object
            prompt: Question or instruction about the image
            instructions: System instructions for the model
            model: Model deployment name (must support vision)
            detail: Image detail level - "low", "high", or "auto"
            media_type: MIME type of the image (default: image/png)
            **kwargs: Additional parameters
            
        Returns:
            ResponseResult with the analysis
            
        Example:
            ```python
            # From file bytes
            with open("chart.png", "rb") as f:
                result = manager.analyze_image(f.read(), "What does this chart show?")
            
            # From URL
            result = manager.analyze_image(
                "https://example.com/image.png",
                "Describe this image"
            )
            ```
        """
        model = model or self.gpt_deployment
        
        # Build image content based on input type
        if isinstance(image, ImageContent):
            image_url = image.to_data_url()
            detail = image.detail
        elif isinstance(image, bytes):
            base64_image = base64.b64encode(image).decode("utf-8")
            image_url = f"data:{media_type};base64,{base64_image}"
        elif isinstance(image, str):
            if image.startswith(("http://", "https://", "data:")):
                image_url = image
            else:
                # Assume it's a base64 string
                image_url = f"data:{media_type};base64,{image}"
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        # Build multimodal message using Responses API format
        # Content types: input_text, input_image, output_text, refusal, input_file, etc.
        user_content = [
            {
                "type": "input_text",
                "text": prompt
            },
            {
                "type": "input_image",
                "image_url": image_url,
                "detail": detail
            }
        ]
        
        # For Responses API, pass instructions separately and only include user message
        request_params = {
            "model": model,
            "input": [{"role": "user", "content": user_content}],
            **kwargs
        }
        
        if instructions:
            request_params["instructions"] = instructions
            
        response = self.client.responses.create(**request_params)
        
        return self._parse_response(response)

    def analyze_multiple_images(
        self,
        images: List[Union[bytes, str, ImageContent]],
        prompt: str,
        instructions: Optional[str] = None,
        model: Optional[str] = None,
        detail: Literal["low", "high", "auto"] = "auto",
        media_type: str = "image/png",
        **kwargs
    ) -> ResponseResult:
        """
        Analyze multiple images in a single request.
        
        Args:
            images: List of images (bytes, base64 strings, URLs, or ImageContent)
            prompt: Question or instruction about the images
            instructions: System instructions for the model
            model: Model deployment name
            detail: Image detail level
            media_type: Default MIME type for byte images
            **kwargs: Additional parameters
            
        Returns:
            ResponseResult with the analysis
        """
        model = model or self.gpt_deployment
        
        # Build content with text first, then all images
        user_content = [{"type": "input_text", "text": prompt}]
        
        for img in images:
            if isinstance(img, ImageContent):
                image_url = img.to_data_url()
                img_detail = img.detail
            elif isinstance(img, bytes):
                base64_image = base64.b64encode(img).decode("utf-8")
                image_url = f"data:{media_type};base64,{base64_image}"
                img_detail = detail
            elif isinstance(img, str):
                if img.startswith(("http://", "https://", "data:")):
                    image_url = img
                else:
                    image_url = f"data:{media_type};base64,{img}"
                img_detail = detail
            else:
                raise ValueError(f"Unsupported image type: {type(img)}")
                
            user_content.append({
                "type": "input_image",
                "image_url": image_url,
                "detail": img_detail
            })
        
        # For Responses API, pass instructions separately and only include user message
        request_params = {
            "model": model,
            "input": [{"role": "user", "content": user_content}],
            **kwargs
        }
        
        if instructions:
            request_params["instructions"] = instructions
            
        response = self.client.responses.create(**request_params)
        
        return self._parse_response(response)

    def generate_embedding(
        self,
        text: Union[str, List[str]],
        model: Optional[str] = None,
    ) -> Union[List[float], List[List[float]]]:
        """
        Generate embeddings for text using the embedding model.
        
        Args:
            text: Single string or list of strings to embed
            model: Embedding model deployment name
            
        Returns:
            Single embedding vector or list of vectors
        """
        model = model or self.embedding_deployment
        
        response = self.client.embeddings.create(
            model=model,
            input=text
        )
        
        if isinstance(text, str):
            return response.data[0].embedding
        return [item.embedding for item in response.data]

    def retrieve_response(self, response_id: str) -> ResponseResult:
        """
        Retrieve a previously created response by ID.
        
        Args:
            response_id: The ID of the response to retrieve
            
        Returns:
            ResponseResult with the response data
        """
        response = self.client.responses.retrieve(response_id)
        return self._parse_response(response)

    def _parse_response(self, response) -> ResponseResult:
        """
        Parse the API response into a standardized ResponseResult.
        
        Args:
            response: Raw API response object
            
        Returns:
            Standardized ResponseResult
        """
        # Extract text content from response output
        content = ""
        if hasattr(response, "output") and response.output:
            for item in response.output:
                if hasattr(item, "type") and item.type == "message":
                    if hasattr(item, "content") and item.content:
                        for content_item in item.content:
                            if hasattr(content_item, "text"):
                                content += content_item.text
        
        # Extract usage information
        usage = {}
        if hasattr(response, "usage") and response.usage:
            usage = {
                "input_tokens": getattr(response.usage, "input_tokens", 0),
                "output_tokens": getattr(response.usage, "output_tokens", 0),
                "total_tokens": getattr(response.usage, "total_tokens", 0),
            }
        
        return ResponseResult(
            content=content,
            response_id=getattr(response, "id", ""),
            model=getattr(response, "model", ""),
            usage=usage,
            raw_response=response
        )

    def stream_complete(
        self,
        prompt: str,
        instructions: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs
    ):
        """
        Stream a text completion response.
        
        Args:
            prompt: The user's input
            instructions: System instructions
            model: Model deployment name
            **kwargs: Additional parameters
            
        Yields:
            Text chunks as they are generated
        """
        model = model or self.gpt_deployment
        
        request_params = {
            "model": model,
            "input": prompt,
            "stream": True,
            **kwargs
        }
        
        if instructions:
            request_params["instructions"] = instructions
            
        response = self.client.responses.create(**request_params)
        
        for event in response:
            if hasattr(event, "type"):
                if event.type == "response.output_text.delta":
                    if hasattr(event, "delta"):
                        yield event.delta
                elif event.type == "response.output_text.done":
                    break
