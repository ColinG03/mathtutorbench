from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Any
import jinja2
import requests
import re
import time
import traceback
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import google.generativeai as genai
from openai import OpenAI
from enum import Enum

# Try to import Tinker SDK for native client (optional)
try:
    import tinker
    TINKER_AVAILABLE = True
except ImportError:
    TINKER_AVAILABLE = False


class ProviderType(Enum):
    COMPLETION_API = "completion_api"
    OLLAMA = "ollama"
    GEMINI = "gemini"


@dataclass
class LLMConfig:
    provider: str  # Will be converted to ProviderType
    model: Optional[str] = None
    base_model: Optional[str] = None  # For Tinker: base model name (e.g., "Qwen/Qwen3-235B")
    base_url: Optional[str] = None
    api_key: Optional[str] = None  # For Gemini
    temperature: float = 0.0
    max_tokens: int = 2048
    is_chat: bool = False
    max_retries: int = 3
    timeout: float = 300.0  # Timeout in seconds for API requests

    def __post_init__(self):
        self.provider = ProviderType(self.provider.lower())
        # Set default base_url for Ollama if not provided
        if self.provider == ProviderType.OLLAMA and not self.base_url:
            raise ValueError("base_url is required for ollama provider")

        # Validate configuration
        # if self.provider == ProviderType.COMPLETION_API and not self.base_url:
        #     raise ValueError("base_url is required for CompletionAPI provider")
        if self.provider == ProviderType.GEMINI and not self.api_key:
            raise ValueError("api_key is required for Gemini provider")


def create_llm_model(config: LLMConfig) -> 'BaseLLMAPI':
    """Factory function to create LLM model instance based on provider"""
    if config.provider == ProviderType.COMPLETION_API:
        return CompletionAPI(config)
    elif config.provider == ProviderType.OLLAMA:
        return OllamaAPI(config)
    elif config.provider == ProviderType.GEMINI:
        return GeminiAPI(config)
    else:
        raise ValueError(f"Unsupported provider: {config.provider}")


class BaseLLMAPI(ABC):
    def __init__(self, config: LLMConfig):
        self.config = config

    def _format_conversation(self, messages: List[Dict]) -> str:
        """Format messages into a conversation string"""
        conversation = ""
        for msg in messages:
            role = msg["user"]
            content = msg["text"]
            conversation += f"{role}: {content}\n"

        return conversation.strip()

    def _format_prompt(self, system_prompt: str, messages: List[Dict]) -> str:
        """Format prompt for non-chat completions"""
        conversation = self._format_conversation(messages)

        if "{{conversation}}" in system_prompt:
            template = jinja2.Template(system_prompt)
            prompt = template.render(conversation=conversation)
        else:
            prompt = system_prompt

        return prompt

    def _format_chat_messages(self, system_prompt: str, messages: List[Dict]) -> List[Dict]:
        """Format messages for chat completions"""
        formatted_messages = [
            {"role": "system", "content": system_prompt}
        ]
        formatted_messages.extend(messages)
        return formatted_messages

    @abstractmethod
    def _make_completion_request(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Make a completion request - to be implemented by specific providers"""
        pass

    @abstractmethod
    def _make_chat_request(self, messages: List[Dict], stop: Optional[List[str]] = None) -> str:
        """Make a chat request - to be implemented by specific providers"""
        pass

    def generate(
            self,
            messages: List[Dict],
            system_prompt: str,
            stop: Optional[List[str]] = None,
    ) -> str:
        """Generate completion using either chat or completion API"""
        print("==============================================================")
        print("Generating completion with model: " + self.config.model)

        try:
            if self.config.is_chat:
                formatted_messages = self._format_chat_messages(system_prompt, messages)
                return self._make_chat_request(formatted_messages, stop)
            else:
                prompt = self._format_prompt(system_prompt, messages)
                return self._make_completion_request(prompt, stop)
        except Exception as e:
            print("Failed to generate completion after retries: " + str(e))
            raise


class CompletionAPI(BaseLLMAPI):
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.tinker_sampling_client = None
        self._tokenizer_cache = None  # Cache tokenizer to avoid repeated calls
        self._tokenizer_lock = threading.Lock()  # Lock for thread-safe tokenizer initialization
        
        # Check if we should use Tinker native client (for models with tinker:// prefix or kimi models)
        use_tinker_native = (
            TINKER_AVAILABLE and 
            config.api_key and 
            config.model and 
            (config.model.startswith('tinker://') or 'kimi' in config.model.lower())
        )
        
        # Determine if we're using a checkpoint path or base model
        # Checkpoint paths contain '/sampler_weights/' or '/checkpoints/'
        is_checkpoint_path = (
            config.model and 
            ('/sampler_weights/' in config.model or '/checkpoints/' in config.model)
        )
        
        # Get base_model from config if explicitly provided, otherwise try to infer
        base_model = getattr(config, 'base_model', None)
        
        if use_tinker_native:
            print(f"Using Tinker native client for model (persistent connection)")
            try:
                # Create Tinker service client and sampling client
                service_client = tinker.ServiceClient(api_key=config.api_key)
                
                # Use checkpoint path if it's a checkpoint, otherwise try base_model
                if is_checkpoint_path or config.model.startswith('tinker://'):
                    # tinker:// paths should use model_path, not base_model
                    # Ensure model path has tinker:// prefix
                    model_path = config.model
                    if not model_path.startswith('tinker://'):
                        model_path = f'tinker://{model_path}'
                    print(f"Using model path: {model_path}")
                    self.tinker_sampling_client = service_client.create_sampling_client(model_path=model_path)
                elif base_model:
                    # base_model should be a HuggingFace model name (e.g., "Qwen/Qwen3-235B")
                    print(f"Using base model: {base_model}")
                    self.tinker_sampling_client = service_client.create_sampling_client(base_model=base_model)
                else:
                    # Assume it's a base model name (HuggingFace format)
                    print(f"Using model name as base model: {config.model}")
                    self.tinker_sampling_client = service_client.create_sampling_client(base_model=config.model)
                
                print(f"[OK] Tinker native client initialized")
                
                # Pre-load tokenizer to avoid concurrent loading issues in batch processing
                try:
                    print("Pre-loading tokenizer...")
                    self._tokenizer_cache = self.tinker_sampling_client.get_tokenizer()
                    print("[OK] Tokenizer pre-loaded")
                except Exception as tokenizer_error:
                    print(f"[WARNING] Could not pre-load tokenizer: {tokenizer_error}")
                    print("Tokenizer will be loaded on first use (may cause delays in batch processing)")
                    
            except Exception as e:
                print(f"[ERROR] Failed to initialize Tinker native client: {e}")
                raise RuntimeError(f"Tinker native client initialization failed: {e}") from e
        
        # Initialize OpenAI client only if NOT using Tinker native client
        if not use_tinker_native:
            if not config.api_key:
                # If no API key provided, try to list available models (for some providers)
                openai_api_key = "EMPTY"
                self.client = OpenAI(
                    api_key=openai_api_key,
                    base_url=config.base_url,
                    timeout=config.timeout,
                )
                try:
                    models = self.client.models.list()
                    print(f"Available models {models}")
                    if hasattr(models, 'data') and models.data:
                        self.config.model = models.data[0].id
                except Exception as e:
                    print(f"Could not list models: {e}")
            else:
                self.client = OpenAI(
                    api_key=config.api_key, 
                    base_url=config.base_url,
                    timeout=config.timeout,
                )
        else:
            # Using Tinker native client - no OpenAI client needed
            self.client = None


    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True
    )
    def _make_completion_request(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        try:
            print("========================(Prompt-completion-start)======================================")
            print(prompt)
            print("========================(Prompt-completion-end)======================================")
            response = self.client.completions.create(
                model=self.config.model,
                prompt=prompt,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                stop=stop
            )
            completion = response.choices[0].text.strip()
            print("===========================(Response-completion-start)===================================")
            print(completion)
            print("===========================(Response-completion-end)===================================")
            return completion
        except Exception as e:
            print("Error in completion request: " + str(e))
            raise

    def _make_tinker_native_request(self, messages: List[Dict], stop: Optional[List[str]] = None) -> str:
        """Use Tinker's native sampling client for persistent connections (better for large models)"""
        try:
            start_time = time.time()
            
            # Create sampling parameters
            # Try to create a proper SamplingParams object, fallback to dict
            try:
                if hasattr(tinker, 'SamplingParams'):
                    sampling_params = tinker.SamplingParams(
                        temperature=self.config.temperature,
                        max_tokens=self.config.max_tokens,
                        stop=stop if stop else None
                    )
                elif hasattr(tinker, 'types') and hasattr(tinker.types, 'SamplingParams'):
                    sampling_params = tinker.types.SamplingParams(
                        temperature=self.config.temperature,
                        max_tokens=self.config.max_tokens,
                        stop=stop if stop else None
                    )
                else:
                    # Fallback to dict
                    sampling_params = {
                        "temperature": self.config.temperature,
                        "max_tokens": self.config.max_tokens,
                    }
                    if stop:
                        sampling_params["stop"] = stop
            except Exception:
                # Fallback to dict if object creation fails
                sampling_params = {
                    "temperature": self.config.temperature,
                    "max_tokens": self.config.max_tokens,
                }
                if stop:
                    sampling_params["stop"] = stop
            
            # Tinker's sample() doesn't accept messages - we need to convert to ModelInput
            # Convert messages to prompt string, then create ModelInput
            result = None
            
            # Convert messages to prompt string
            prompt_parts = []
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "system":
                    prompt_parts.append(f"System: {content}")
                elif role == "user":
                    prompt_parts.append(f"User: {content}")
                elif role == "assistant":
                    prompt_parts.append(f"Assistant: {content}")
            
            prompt_text = "\n".join(prompt_parts)
            if not prompt_text.endswith("\nAssistant:"):
                prompt_text += "\nAssistant:"
            
            # Create ModelInput from text using tokenizer
            try:
                # Use cached tokenizer or get it once (thread-safe)
                if self._tokenizer_cache is None:
                    with self._tokenizer_lock:
                        # Double-check pattern to avoid race conditions
                        if self._tokenizer_cache is None:
                            self._tokenizer_cache = self.tinker_sampling_client.get_tokenizer()
                tokenizer = self._tokenizer_cache
                
                # Encode text to tokens
                tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
                
                # Create EncodedTextChunk
                text_chunk = tinker.EncodedTextChunk(tokens=tokens)
                
                # Create ModelInput with the chunk
                model_input = tinker.ModelInput(chunks=[text_chunk])
                
                # Now call sample with ModelInput
                result = self.tinker_sampling_client.sample(
                    prompt=model_input,
                    sampling_params=sampling_params,
                    num_samples=1
                )
            except Exception as e2:
                import traceback
                traceback.print_exc()
                raise
            
            if result is None:
                raise ValueError("Could not determine correct Tinker sample() API signature")
            
            # Tinker's sample() returns a ConcurrentFuture - we need to wait for it
            # Check if it has a result() method (works for both Future and ConcurrentFuture)
            if hasattr(result, 'result'):
                # Check if already done to avoid unnecessary waiting
                if hasattr(result, 'done') and result.done():
                    try:
                        result = result.result()  # Get result without timeout if already done
                    except Exception as e:
                        elapsed = time.time() - start_time
                        raise RuntimeError(f"Future completed with error after {elapsed:.2f}s: {type(e).__name__}: {e}") from e
                else:
                    # Wait for the future to complete (with timeout)
                    # For large models, use a longer timeout (10 minutes)
                    timeout = max(self.config.timeout, 600.0)  # At least 10 minutes for large models
                    try:
                        result = result.result(timeout=timeout)
                    except TimeoutError as e:
                        elapsed = time.time() - start_time
                        raise RuntimeError(f"Request timed out after {elapsed:.2f}s (timeout={timeout}s). The model may be overloaded or the request is too complex.") from e
                    except Exception as e:
                        elapsed = time.time() - start_time
                        raise RuntimeError(f"Error waiting for Future after {elapsed:.2f}s: {type(e).__name__}: {e}") from e
            
            elapsed_time = time.time() - start_time
            print(f"Response received in {elapsed_time:.2f}s")
            
            # Extract the completion text from SampleResponse
            completion = None
            
            # Try to get text from sequences (most common case)
            if hasattr(result, 'sequences') and len(result.sequences) > 0:
                sequence = result.sequences[0]
                # Try to get text directly from sequence
                if hasattr(sequence, 'text'):
                    completion = sequence.text
                elif hasattr(sequence, 'completion'):
                    completion = sequence.completion
                elif hasattr(sequence, 'content'):
                    completion = sequence.content
                elif hasattr(sequence, 'tokens') and sequence.tokens:
                    # Decode tokens using cached tokenizer
                    if self._tokenizer_cache is None:
                        self._tokenizer_cache = self.tinker_sampling_client.get_tokenizer()
                    completion = self._tokenizer_cache.decode(sequence.tokens, skip_special_tokens=True)
            
            # Fallback to other attributes
            if not completion:
                if hasattr(result, 'samples') and len(result.samples) > 0:
                    sample = result.samples[0]
                    if hasattr(sample, 'text'):
                        completion = sample.text
                    elif hasattr(sample, 'completion'):
                        completion = sample.completion
                    elif hasattr(sample, 'content'):
                        completion = sample.content
                elif hasattr(result, 'text'):
                    completion = result.text
                elif hasattr(result, 'completion'):
                    completion = result.completion
                elif hasattr(result, 'content'):
                    completion = result.content
            
            # Last resort: decode from sequences if we have tokens
            if not completion and hasattr(result, 'sequences') and len(result.sequences) > 0:
                sequence = result.sequences[0]
                if hasattr(sequence, 'tokens') and sequence.tokens:
                    if self._tokenizer_cache is None:
                        self._tokenizer_cache = self.tinker_sampling_client.get_tokenizer()
                    completion = self._tokenizer_cache.decode(sequence.tokens, skip_special_tokens=True)
            
            if not completion:
                raise ValueError(f"Could not extract text from SampleResponse. Available attributes: {[attr for attr in dir(result) if not attr.startswith('_')]}")
            
            # Extract thinking tokens if present
            if self.config.model and ('kimi' in self.config.model.lower() or 'qwen' in self.config.model.lower()):
                thinking_pattern = r'<think>(.*?)</think>'
                thinking_tokens = re.findall(thinking_pattern, completion, re.DOTALL)
                
                if thinking_tokens:
                    print("===========================(Thinking-tokens-start)===================================")
                    for i, thinking in enumerate(thinking_tokens, 1):
                        print(f"Thinking token {i}:")
                        print(thinking)
                    print("===========================(Thinking-tokens-end)===================================")
                    completion = re.sub(thinking_pattern, '', completion, flags=re.DOTALL).strip()
            
            print("===========================(Response-chat-start)===================================")
            print(completion)
            print("===========================(Response-chat-end)===================================")
            return completion
            
        except Exception as e:
            print(f"Error in Tinker native client: {type(e).__name__}: {e}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=4, max=30),
        retry=retry_if_exception_type((Exception,)),
        reraise=True
    )
    def _make_chat_request(self, messages: List[Dict], stop: Optional[List[str]] = None) -> str:
        try:
            # Use Tinker native client if available (for Kimi and other large models)
            if self.tinker_sampling_client:
                return self._make_tinker_native_request(messages, stop)
            
            # If using Tinker native client, we shouldn't reach here
            if self.client is None:
                raise RuntimeError("Tinker native client not initialized and OpenAI client is None")
            
            print("========================(Prompt-chat-start)======================================")
            print(messages[0]["content"])
            print("========================(Prompt-chat-end)======================================")
            
            # For thinking models, add thinking-specific stop sequences if not already present
            stop_sequences = list(stop) if stop else []
            if self.config.model and ('kimi' in self.config.model.lower() or 'thinking' in self.config.model.lower()):
                thinking_stops = ['</think>', '\n\n\n', '\n\nProblem:']
                for ts in thinking_stops:
                    if ts not in stop_sequences:
                        stop_sequences.append(ts)
            
            start_time = time.time()
            
            try:
                response = self.client.chat.completions.create(
                    model=self.config.model,
                    messages=messages,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    stop=stop_sequences if stop_sequences else None,
                    stream=False
                )
                elapsed_time = time.time() - start_time
                print(f"Response received in {elapsed_time:.2f}s")
                completion = response.choices[0].message.content
                
                # Extract thinking tokens for Kimi/Qwen
                if self.config.model and ('kimi' in self.config.model.lower() or 'qwen' in self.config.model.lower()):
                    thinking_pattern = r'<think>(.*?)</think>'
                    thinking_tokens = re.findall(thinking_pattern, completion, re.DOTALL)
                    
                    if thinking_tokens:
                        print("===========================(Thinking-tokens-start)===================================")
                        for i, thinking in enumerate(thinking_tokens, 1):
                            print(f"Thinking token {i}:")
                            print(thinking)
                        print("===========================(Thinking-tokens-end)===================================")
                        completion = re.sub(thinking_pattern, '', completion, flags=re.DOTALL).strip()
                
                print("===========================(Response-chat-start)===================================")
                print(completion)
                print("===========================(Response-chat-end)===================================")
                return completion
                
            except Exception as api_error:
                elapsed_time = time.time() - start_time
                print(f"Error in API call after {elapsed_time:.2f}s: {type(api_error).__name__}: {api_error}")
                raise
                
        except Exception as e:
            print(f"Error in chat request: {type(e).__name__}: {e}")
            raise

    def generate_batch(
        self,
        batch_items: List[Dict[str, Any]],
        max_workers: int = 5
    ) -> List[Optional[str]]:
        """Generate completions for multiple items concurrently"""
        results = {}
        
        def generate_one(item_idx, item):
            try:
                return item_idx, self.generate(
                    messages=item["messages"],
                    system_prompt=item["system_prompt"],
                    stop=item.get("stop")
                )
            except Exception as e:
                print(f"Error in batch item {item_idx}: {e}")
                return item_idx, None
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(generate_one, idx, item): idx 
                for idx, item in enumerate(batch_items)
            }
            
            for future in as_completed(futures):
                item_idx, result = future.result()
                results[item_idx] = result
        
        # Return results in original order
        return [results[i] for i in sorted(results.keys())]


class OllamaAPI(BaseLLMAPI):
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.base_url = config.base_url

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True
    )
    def _make_completion_request(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Make a completion request to Ollama"""
        try:
            print("========================(Prompt-start)======================================")
            print(prompt)
            print("========================(Prompt-end)======================================")
            response = requests.post(
                f"{self.base_url}",
                json={
                    "model": self.config.model,
                    "prompt": prompt,
                    "options": {
                        "temperature": self.config.temperature,
                        "num_predict": self.config.max_tokens,
                        "stop": stop or []
                    },
                    "stream": False
                }
            )
            response.raise_for_status()
            completion = response.json()["response"].strip()
            print("===========================(Response-start)===================================")
            print(completion)
            print("===========================(Response-end)===================================")
            return completion
        except Exception as e:
            print("Error in completion request: " + str(e))
            raise

    def _make_chat_request(self, messages: List[Dict], stop: Optional[List[str]] = None) -> str:
        """Make a chat request to Ollama"""
        # Ollama doesn't have a separate chat endpoint, so we'll format messages into a prompt
        formatted_prompt = ""
        for message in messages:
            formatted_prompt += f"{message['role']}: {message['content']}\nassistant: "

        return self._make_completion_request(formatted_prompt, stop)


class GeminiAPI(BaseLLMAPI):
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        genai.configure(api_key=config.api_key)
        self.model = genai.GenerativeModel(config.model)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True
    )
    def _make_completion_request(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        try:
            print("========================(Prompt-start)======================================")
            print(prompt)
            print("========================(Prompt-end)======================================")
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=self.config.temperature,
                    max_output_tokens=self.config.max_tokens,
                    stop_sequences=stop or []
                )
            )
            completion = response.text
            print("===========================(Response-start)===================================")
            print(completion)
            print("===========================(Response-end)===================================")
            return completion
        except Exception as e:
            print("Error in completion request: " + str(e))
            raise

    def _make_chat_request(self, messages: List[Dict], stop: Optional[List[str]] = None) -> str:
        print("========================(Prompt-chat-start)======================================")
        print(messages[0]["content"])
        print("========================(Prompt-chat-end)======================================")
        chat = self.model.start_chat()
        if messages[0]["role"] == "system":
            # Add system prompt as first user message
            response = chat.send_message(messages[0]["content"], stream=False,
                                         generation_config=genai.types.GenerationConfig(
                                             temperature=self.config.temperature,
                                             max_output_tokens=self.config.max_tokens,
                                             stop_sequences=stop or []
                                         ))
        else:
            # chat.send_message(message["content"], role=message["role"])
            raise ValueError("No system prompt")

        completion = response.text
        print("===========================(Response-chat-start)===================================")
        print(completion)
        print("===========================(Response-chat-end)===================================")
        return completion
