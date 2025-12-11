"""
Model Interface - Provider-agnostic LLM communication layer
"""
import os
import re
import json
import logging
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class ModelInterface:
    """
    Unified interface for interacting with language model providers.
    
    Configuration via environment variables:
    - LLM_PROVIDER: anthropic|local|google|ollama
    - LLM_MODEL: specific model identifier (e.g., llama3.2)
    - LLM_API_KEY: authentication key
    - LLM_TEMPERATURE: sampling temperature (default: 0.3)
    - LLM_BASE_URL: base URL for local providers
    """
    
    def __init__(self):
        self.provider = os.getenv("LLM_PROVIDER", "google").lower()
        self.model_name = os.getenv("LLM_MODEL", "gemini-2.5-flash-lite")
        self.api_key = os.getenv("LLM_API_KEY") or os.getenv("GEMINI_API_KEY")
        self.default_temperature = float(os.getenv("LLM_TEMPERATURE", "0.3"))
        
        self._initialize_provider()
    
    def _initialize_provider(self):
        """Initialize the appropriate provider client."""
        try:
            if self.provider == "google":
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self.client = genai.GenerativeModel(self.model_name)
                logger.info(f"Initialized Google provider with model: {self.model_name}")
            
            elif self.provider == "anthropic":
                try:
                    from anthropic import Anthropic
                    self.client = Anthropic(api_key=self.api_key)
                    logger.info(f"Initialized Anthropic provider with model: {self.model_name}")
                except ImportError:
                    raise ImportError("Anthropic SDK not installed. Run: pip install anthropic")
            
            elif self.provider == "ollama":
                # Ollama uses the OpenAI SDK format
                from openai import OpenAI
                # Default Ollama URL is localhost:11434. The /v1 is required for compatibility.
                endpoint_url = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
                # Ollama does not require a real API key, but the SDK requires a non-empty string
                self.client = OpenAI(api_key="ollama", base_url=endpoint_url)
                logger.info(f"Initialized Ollama provider at: {endpoint_url} with model: {self.model_name}")

            elif self.provider == "local":
                from openai import OpenAI
                endpoint_url = os.getenv("LLM_BASE_URL", "http://localhost:8000/v1")
                self.client = OpenAI(api_key="not-needed", base_url=endpoint_url)
                logger.info(f"Initialized local provider at: {endpoint_url}")
            
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
        
        except ImportError as e:
            logger.error(f"Failed to import required library: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Provider initialization failed: {str(e)}")
            raise
    
    def generate_json(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generate structured JSON response from model.
        
        Args:
            messages: Conversation history
            temperature: Sampling temperature override
            max_tokens: Maximum response length
            
        Returns:
            Parsed JSON dictionary
        """
        message_list = messages.copy()
        if message_list[-1]["role"] == "user":
            message_list[-1]["content"] += "\n\nIMPORTANT: You MUST respond with ONLY valid JSON. No explanations, no markdown, no code blocks. Just raw JSON starting with { and ending with }."
        
        response_text = self.generate(
            messages=message_list,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        if not response_text or not response_text.strip():
            logger.error("Empty response from model")
            raise ValueError("Model returned empty response")
        
        # Strip markdown code blocks
        cleaned_text = response_text.strip()
        if cleaned_text.startswith("```"):
            # Remove opening ```json or ```
            lines = cleaned_text.split('\n')
            if lines[0].startswith("```"):
                lines = lines[1:]
            # Remove closing ```
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            cleaned_text = '\n'.join(lines).strip()
        
        logger.debug(f"Raw response: {response_text[:500]}")
        logger.debug(f"Cleaned response: {cleaned_text[:500]}")
        
        try:
            return json.loads(cleaned_text)
        except json.JSONDecodeError as e:
            logger.warning(f"Initial JSON parse failed: {str(e)}")
            logger.warning(f"Response text: {cleaned_text[:300]}")
            
            json_pattern = re.search(r'\{.*\}', cleaned_text, re.DOTALL)
            if json_pattern:
                try:
                    extracted = json_pattern.group()
                    return json.loads(extracted)
                except json.JSONDecodeError as parse_err:
                    logger.error(f"Failed to parse extracted JSON: {str(parse_err)}")
                    logger.error(f"Extracted text: {extracted[:300]}")
                    raise ValueError(f"Invalid JSON in response: {cleaned_text[:200]}")
            
            logger.error(f"No JSON object found in response: {cleaned_text[:300]}")
            raise ValueError(f"No JSON found in response: {cleaned_text[:200]}")
    
    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Generate text completion from model.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens in response
            response_format: Format specification for structured output
            
        Returns:
            Generated text response
        """
        temp = temperature if temperature is not None else self.default_temperature
        token_limit = max_tokens or 2000
        
        try:
            if self.provider == "google":
                return self._generate_google(messages, temp, token_limit)
            elif self.provider == "anthropic":
                return self._generate_anthropic(messages, temp, token_limit)
            elif self.provider == "local" or self.provider == "ollama":
                return self._generate_local(messages, temp, token_limit, response_format)
            else:
                raise NotImplementedError(f"Generation not supported for: {self.provider}")
        
        except Exception as e:
            logger.error(f"Generation failed for provider {self.provider}: {str(e)}")
            raise
    
    def _generate_google(self, messages: List[Dict[str, str]], temp: float, token_limit: int) -> str:
        """Generate response using Google Gemini."""
        prompt_text = ""
        for message in messages:
            role_label = message["role"]
            message_content = message["content"]
            
            if role_label == "system":
                prompt_text += f"System: {message_content}\n"
            elif role_label == "user":
                prompt_text += f"User: {message_content}\n"
            elif role_label == "assistant":
                prompt_text += f"Assistant: {message_content}\n"
        
        config = {
            "temperature": temp,
            "max_output_tokens": token_limit,
        }
        
        try:
            result = self.client.generate_content(
                prompt_text,
                generation_config=config
            )
            
            if not result or not hasattr(result, 'text'):
                logger.error("Invalid response structure from Google API")
                raise ValueError("No text in Google API response")
            
            response_text = result.text
            if not response_text:
                logger.error("Empty text from Google API")
                raise ValueError("Google API returned empty text")
            
            return response_text
        except AttributeError as e:
            logger.error(f"Google API response missing text attribute: {str(e)}")
            logger.error(f"Response object: {result}")
            raise ValueError("Invalid response from Google API")
        except Exception as e:
            logger.error(f"Google API error: {str(e)}")
            raise
    
    def _generate_anthropic(self, messages: List[Dict[str, str]], temp: float, token_limit: int) -> str:
        """Generate response using Anthropic Claude."""
        system_content = None
        conversation_messages = []
        
        for message in messages:
            if message["role"] == "system":
                system_content = message["content"]
            else:
                conversation_messages.append(message)
        
        request_params = {
            "model": self.model_name,
            "messages": conversation_messages,
            "temperature": temp,
            "max_tokens": token_limit
        }
        
        if system_content:
            request_params["system"] = system_content
        
        try:
            result = self.client.messages.create(**request_params)
            return result.content[0].text
        except Exception as e:
            logger.error(f"Anthropic API error: {str(e)}")
            raise
    
    def _generate_local(
        self,
        messages: List[Dict[str, str]],
        temp: float,
        token_limit: int,
        response_format: Optional[Dict[str, str]]
    ) -> str:
        """Generate response using local model server (or Ollama)."""
        request_params = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temp,
            "max_tokens": token_limit
        }
        
        if response_format:
            request_params["response_format"] = response_format
        
        try:
            result = self.client.chat.completions.create(**request_params)
            return result.choices[0].message.content
        except Exception as e:
            logger.error(f"Local/Ollama server error: {str(e)}")
            raise


_model_interface_instance = None


def get_llm_client() -> ModelInterface:
    """Retrieve or instantiate the singleton model interface."""
    global _model_interface_instance
    if _model_interface_instance is None:
        _model_interface_instance = ModelInterface()
    return _model_interface_instance


def load_prompt(filename: str) -> str:
    """Load prompt template from prompts directory."""
    base_dir = os.path.dirname(os.path.dirname(__file__))
    prompt_file_path = os.path.join(base_dir, "prompts", filename)
    
    try:
        with open(prompt_file_path, "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        logger.error(f"Prompt file not found: {prompt_file_path}")
        raise
    except Exception as e:
        logger.error(f"Failed to load prompt file: {str(e)}")
        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    try:
        # Example: Set ENV vars programmatically for testing, or rely on .env
        # os.environ["LLM_PROVIDER"] = "ollama"
        # os.environ["LLM_MODEL"] = "llama3.2"
        
        interface = get_llm_client()
        logger.info(f"Model interface ready - Provider: {interface.provider}")
        
        test_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say hello!"}
        ]
        
        logger.info("Sending test request...")
        response = interface.generate(test_messages)
        logger.info(f"Response received: {response}")
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")