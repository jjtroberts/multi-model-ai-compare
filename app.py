import asyncio
import aiohttp
import streamlit as st
from dataclasses import dataclass
from typing import List, Dict, Optional, Any
import json
import time
import os
from datetime import datetime
import logging
import traceback
import re

# Enhanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load API keys from environment variables
def load_env_api_keys():
    """Load API keys from environment variables"""
    return {
        'claude': os.getenv('ANTHROPIC_API_KEY', ''),
        'chatgpt': os.getenv('OPENAI_API_KEY', ''), 
        'gemini': os.getenv('GOOGLE_API_KEY', '')
    }

@dataclass
class DetailedError:
    """Enhanced error reporting structure"""
    error_type: str
    error_message: str
    http_status: Optional[int] = None
    response_headers: Optional[Dict] = None
    request_url: Optional[str] = None
    request_payload: Optional[Dict] = None
    raw_response: Optional[str] = None
    troubleshooting_tips: List[str] = None
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
        if not self.troubleshooting_tips:
            self.troubleshooting_tips = []

@dataclass
class ModelResponse:
    model_name: str
    response: str
    tokens_used: Optional[int] = None
    response_time: float = 0.0
    error: Optional[str] = None
    detailed_error: Optional[DetailedError] = None

class GeminiDebugger:
    """Enhanced debugging and error reporting for Gemini API"""
    
    def __init__(self):
        self.debug_mode = True
        self.request_history = []
    
    def validate_api_key(self, api_key: str) -> tuple[bool, List[str]]:
        """Validate Gemini API key format and provide troubleshooting tips"""
        issues = []
        
        if not api_key:
            issues.append("API key is empty")
            return False, issues
        
        # Check basic format (Gemini keys typically start with specific patterns)
        if not re.match(r'^[A-Za-z0-9_-]+$', api_key):
            issues.append("API key contains invalid characters")
        
        if len(api_key) < 20:
            issues.append("API key appears too short (typical keys are 39+ characters)")
        
        if len(api_key) > 100:
            issues.append("API key appears too long")
        
        # Check for common mistakes
        if api_key.startswith('sk-'):
            issues.append("This appears to be an OpenAI key (starts with 'sk-'), not a Google API key")
        
        if api_key.startswith('claude-'):
            issues.append("This appears to be a Claude key format, not a Google API key")
        
        return len(issues) == 0, issues
    
    def get_troubleshooting_tips(self, error_type: str, status_code: Optional[int] = None) -> List[str]:
        """Get specific troubleshooting tips based on error type"""
        tips = []
        
        if status_code == 400:
            tips.extend([
                "🔧 Check if the model name is correct and available",
                "🔧 Verify the request payload format matches Gemini API specs",
                "🔧 Ensure maxOutputTokens is within allowed range (1-8192)",
                "🔧 Check if the prompt text contains unsupported characters"
            ])
        elif status_code == 401:
            tips.extend([
                "🔑 Verify your API key is correct and active",
                "🔑 Check if the API key has proper permissions",
                "🔑 Ensure you're using a Google AI Studio API key, not other Google service keys"
            ])
        elif status_code == 403:
            tips.extend([
                "🚫 Check if Gemini API is enabled in your Google Cloud project",
                "🚫 Verify your account has access to the requested model",
                "🚫 Check regional availability of the model",
                "🚫 Ensure you're not exceeding quota limits"
            ])
        elif status_code == 404:
            tips.extend([
                "🔍 Verify the model name exists and is spelled correctly",
                "🔍 Check if the model is available in your region",
                "🔍 Try using a different model version (e.g., gemini-1.5-pro vs gemini-1.5-pro-latest)"
            ])
        elif status_code == 429:
            tips.extend([
                "⏱️ You're being rate limited - wait before retrying",
                "⏱️ Consider upgrading your API plan for higher rate limits",
                "⏱️ Implement exponential backoff in your requests"
            ])
        elif status_code == 500:
            tips.extend([
                "🔄 Google's servers are experiencing issues - try again later",
                "🔄 Try a different model if available",
                "🔄 Reduce the complexity or length of your prompt"
            ])
        elif error_type == "connection":
            tips.extend([
                "🌐 Check your internet connection",
                "🌐 Verify firewall settings allow HTTPS to generativelanguage.googleapis.com",
                "🌐 Try using a different DNS server"
            ])
        elif error_type == "timeout":
            tips.extend([
                "⏰ Request timed out - try reducing prompt length",
                "⏰ Reduce maxOutputTokens to speed up response",
                "⏰ Check your network stability"
            ])
        
        # General tips for all errors
        if not tips:
            tips.extend([
                "🔧 Check the Gemini API documentation: https://ai.google.dev/docs",
                "🔧 Verify your API key at https://aistudio.google.com/app/apikey",
                "🔧 Try a simple test request with curl to isolate the issue"
            ])
        
        return tips
    
    def log_request_details(self, url: str, headers: Dict, payload: Dict, api_key: str):
        """Log detailed request information for debugging"""
        masked_key = f"{api_key[:8]}...{api_key[-4:]}" if len(api_key) > 12 else "***"
        
        logger.info(f"🚀 Gemini API Request:")
        logger.info(f"   URL: {url}")
        logger.info(f"   API Key: {masked_key}")
        logger.info(f"   Payload: {json.dumps(payload, indent=2)}")
        logger.info(f"   Headers: {json.dumps({k: v for k, v in headers.items() if k.lower() != 'authorization'}, indent=2)}")
    
    def log_response_details(self, status: int, headers: Dict, response_text: str):
        """Log detailed response information for debugging"""
        logger.info(f"📥 Gemini API Response:")
        logger.info(f"   Status: {status}")
        logger.info(f"   Headers: {json.dumps(dict(headers), indent=2)}")
        logger.info(f"   Response length: {len(response_text)} characters")
        logger.info(f"   Response preview: {response_text[:500]}...")

class ModelDiscovery:
    """Dynamically discover available models from API providers"""
    
    def __init__(self):
        self.cache = {}
        self.cache_duration = 300  # 5 minutes
        
    async def get_claude_models(self, api_key: str) -> List[Dict]:
        """Fetch available Claude models from Anthropic API (free)"""
        try:
            headers = {
                'x-api-key': api_key,
                'anthropic-version': '2023-06-01'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get('https://api.anthropic.com/v1/models', headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        # Models are returned newest first
                        models = []
                        for model in data.get('data', []):
                            models.append({
                                'id': model['id'],
                                'name': model.get('display_name', model['id']),
                                'created': model.get('created_at', '')
                            })
                        return models
                    else:
                        logger.warning(f"Claude models API returned {response.status}")
                        return []
        except Exception as e:
            logger.warning(f"Failed to fetch Claude models: {e}")
            return []
    
    async def get_openai_models(self, api_key: str) -> List[Dict]:
        """Fetch available OpenAI models from API (free)"""
        try:
            headers = {
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get('https://api.openai.com/v1/models', headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        models = []
                        # Filter for chat models and sort by creation date
                        chat_models = [m for m in data.get('data', []) if 'gpt' in m['id'].lower()]
                        chat_models.sort(key=lambda x: x.get('created', 0), reverse=True)
                        
                        for model in chat_models:
                            model_id = model['id']
                            # Skip fine-tuned models and old models
                            if ':' not in model_id and not model_id.startswith('text-'):
                                models.append({
                                    'id': model_id,
                                    'name': model_id.upper().replace('-', ' '),
                                    'created': model.get('created', 0)
                                })
                        return models
                    else:
                        logger.warning(f"OpenAI models API returned {response.status}")
                        return []
        except Exception as e:
            logger.warning(f"Failed to fetch OpenAI models: {e}")
            return []
    
    async def get_gemini_models(self, api_key: str) -> List[Dict]:
        """Fetch available Gemini models from Google API (free) with enhanced debugging"""
        debugger = GeminiDebugger()
        
        try:
            url = f'https://generativelanguage.googleapis.com/v1beta/models?key={api_key}'
            headers = {'User-Agent': 'Multi-Model-AI-Comparison/1.0'}
            
            # Log request details
            debugger.log_request_details(url, headers, {}, api_key)
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    response_text = await response.text()
                    
                    # Log response details
                    debugger.log_response_details(response.status, response.headers, response_text)
                    
                    if response.status == 200:
                        data = await response.json()
                        logger.info(f"Gemini models API response keys: {list(data.keys())}")
                        
                        models = []
                        models_list = data.get('models', [])
                        
                        if not models_list:
                            logger.warning("No models found in Gemini API response")
                            logger.info(f"Full response: {json.dumps(data, indent=2)}")
                            return []
                        
                        for model in models_list:
                            model_name = model.get('name', '').replace('models/', '')
                            # Only include generative models
                            supported_methods = model.get('supportedGenerationMethods', [])
                            if 'generateContent' in supported_methods:
                                # Create a better display name
                                display_name = model_name.replace('-', ' ').replace('_', ' ').title()
                                models.append({
                                    'id': model_name,
                                    'name': display_name,
                                    'created': model.get('createTime', '')
                                })
                        
                        # Sort by name to get latest versions first (2.0, 1.5, etc.)
                        models.sort(key=lambda x: x['name'], reverse=True)
                        logger.info(f"Found {len(models)} Gemini models")
                        return models
                    else:
                        logger.error(f"Gemini models API returned {response.status}: {response_text}")
                        return []
        except Exception as e:
            logger.error(f"Failed to fetch Gemini models: {e}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
            return []
    
    def get_fallback_models(self, provider: str) -> Dict[str, str]:
        """Fallback static model lists if API fails"""
        fallbacks = {
            'claude': {
                'Claude Opus 4': 'claude-opus-4-20250514',
                'Claude Sonnet 4': 'claude-sonnet-4-20250514',
                'Claude 3.5 Sonnet (Latest)': 'claude-3-5-sonnet-20241022',
                'Claude 3.5 Haiku': 'claude-3-5-haiku-20241022'
            },
            'chatgpt': {
                'GPT-4o': 'gpt-4o',
                'GPT-4o mini': 'gpt-4o-mini',
                'GPT-4 Turbo': 'gpt-4-turbo',
                'GPT-4': 'gpt-4'
            },
            'gemini': {
                'Gemini 1.5 Pro': 'gemini-1.5-pro-latest',
                'Gemini 1.5 Flash': 'gemini-1.5-flash-latest',
                'Gemini 1.5 Pro (Stable)': 'gemini-1.5-pro',
                'Gemini 1.5 Flash (Stable)': 'gemini-1.5-flash',
                'Gemini 1.0 Pro': 'gemini-pro'
            }
        }
        return fallbacks.get(provider, {})

class MultiModelAgent:
    def __init__(self):
        self.discovery = ModelDiscovery()
        self.gemini_debugger = GeminiDebugger()
        # Simplified API configurations (URLs and headers only)
        self.apis = {
            'claude': {
                'url': 'https://api.anthropic.com/v1/messages',
                'headers': lambda key: {
                    'Content-Type': 'application/json',
                    'x-api-key': key,
                    'anthropic-version': '2023-06-01'
                }
            },
            'chatgpt': {
                'url': 'https://api.openai.com/v1/chat/completions',
                'headers': lambda key: {
                    'Content-Type': 'application/json',
                    'Authorization': f'Bearer {key}'
                }
            },
            'gemini': {
                'url': 'https://generativelanguage.googleapis.com/v1beta/models',
                'headers': lambda key: {
                    'Content-Type': 'application/json',
                }
            }
        }
    
    async def get_available_models(self, provider: str, api_key: str) -> tuple[Dict[str, str], bool]:
        """
        Get available models for a provider.
        Returns (models_dict, is_dynamic) where is_dynamic indicates if models were fetched from API
        """
        if not api_key:
            return self.discovery.get_fallback_models(provider), False
        
        # Check cache first
        cache_key = f"{provider}_{api_key[:8]}"
        cached = self.discovery.cache.get(cache_key)
        if cached and time.time() - cached['timestamp'] < self.discovery.cache_duration:
            return cached['models'], cached['is_dynamic']
        
        try:
            # Fetch models dynamically
            if provider == 'claude':
                models_list = await self.discovery.get_claude_models(api_key)
            elif provider == 'chatgpt':
                models_list = await self.discovery.get_openai_models(api_key)
            elif provider == 'gemini':
                models_list = await self.discovery.get_gemini_models(api_key)
            else:
                models_list = []
            
            if models_list:
                # Convert to dict format
                models_dict = {model['name']: model['id'] for model in models_list}
                # Cache the result
                self.discovery.cache[cache_key] = {
                    'models': models_dict,
                    'is_dynamic': True,
                    'timestamp': time.time()
                }
                return models_dict, True
            else:
                # Fallback to static list
                fallback = self.discovery.get_fallback_models(provider)
                self.discovery.cache[cache_key] = {
                    'models': fallback,
                    'is_dynamic': False,
                    'timestamp': time.time()
                }
                return fallback, False
                
        except Exception as e:
            logger.warning(f"Failed to fetch {provider} models: {e}")
            fallback = self.discovery.get_fallback_models(provider)
            return fallback, False
    
    async def query_claude(self, prompt: str, api_key: str, model: str, max_tokens: int = 2000) -> ModelResponse:
        start_time = time.time()
        try:
            payload = {
                "model": model,
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}]
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.apis['claude']['url'],
                    headers=self.apis['claude']['headers'](api_key),
                    json=payload
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        # Extract model name from the full model string  
                        model_display = model  # Use the actual model ID since we may not have a static mapping
                        return ModelResponse(
                            model_name=f"Claude ({model_display})",
                            response=data['content'][0]['text'],
                            tokens_used=data.get('usage', {}).get('output_tokens'),
                            response_time=time.time() - start_time
                        )
                    else:
                        error_text = await response.text()
                        return ModelResponse(
                            model_name=f"Claude ({model})",
                            response="",
                            error=f"HTTP {response.status}: {error_text}",
                            response_time=time.time() - start_time
                        )
        except Exception as e:
            return ModelResponse(
                model_name=f"Claude ({model})",
                response="",
                error=str(e),
                response_time=time.time() - start_time
            )
    
    async def query_chatgpt(self, prompt: str, api_key: str, model: str, max_tokens: int = 2000) -> ModelResponse:
        start_time = time.time()
        try:
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.apis['chatgpt']['url'],
                    headers=self.apis['chatgpt']['headers'](api_key),
                    json=payload
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return ModelResponse(
                            model_name=f"ChatGPT ({model})",
                            response=data['choices'][0]['message']['content'],
                            tokens_used=data.get('usage', {}).get('completion_tokens'),
                            response_time=time.time() - start_time
                        )
                    else:
                        error_text = await response.text()
                        error_msg = f"HTTP {response.status}: {error_text}"
                        
                        # Add helpful context for common errors
                        if response.status == 401:
                            error_msg += "\n💡 Tip: Check your OpenAI API key is valid"
                        elif response.status == 429:
                            error_msg += "\n💡 Tip: Rate limited - you may need to upgrade your OpenAI plan"
                        elif response.status == 404:
                            error_msg += "\n💡 Tip: Model may require higher OpenAI API tier"
                        
                        return ModelResponse(
                            model_name=f"ChatGPT ({model})",
                            response="",
                            error=error_msg,
                            response_time=time.time() - start_time
                        )
        except Exception as e:
            return ModelResponse(
                model_name=f"ChatGPT ({model})",
                response="",
                error=str(e),
                response_time=time.time() - start_time
            )
    
    async def query_gemini(self, prompt: str, api_key: str, model: str, max_tokens: int = 2000) -> ModelResponse:
        """Enhanced Gemini query with detailed error reporting"""
        start_time = time.time()
        debugger = self.gemini_debugger
        
        try:
            # Validate API key first
            is_valid, validation_issues = debugger.validate_api_key(api_key)
            if not is_valid:
                detailed_error = DetailedError(
                    error_type="validation",
                    error_message="API key validation failed",
                    troubleshooting_tips=validation_issues + [
                        "🔑 Get a valid API key from https://aistudio.google.com/app/apikey",
                        "🔑 Ensure you're using Google AI Studio API key, not Google Cloud API key"
                    ]
                )
                return ModelResponse(
                    model_name=f"Gemini ({model})",
                    response="",
                    error=f"API key validation failed: {'; '.join(validation_issues)}",
                    detailed_error=detailed_error,
                    response_time=time.time() - start_time
                )
            
            # Construct URL and payload
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "maxOutputTokens": max_tokens,
                    "temperature": 0.7
                }
            }
            headers = {
                'Content-Type': 'application/json',
                'User-Agent': 'Multi-Model-AI-Comparison/1.0'
            }
            
            # Log request details for debugging
            if debugger.debug_mode:
                debugger.log_request_details(url, headers, payload, api_key)
            
            # Make the request with timeout
            timeout = aiohttp.ClientTimeout(total=60, connect=30)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    response_text = await response.text()
                    response_headers = dict(response.headers)
                    
                    # Log response details
                    if debugger.debug_mode:
                        debugger.log_response_details(response.status, response_headers, response_text)
                    
                    if response.status == 200:
                        try:
                            data = json.loads(response_text)
                            
                            # Validate response structure
                            if 'candidates' not in data:
                                raise ValueError(f"Unexpected response structure: {data}")
                            
                            if not data['candidates']:
                                raise ValueError("No candidates in response")
                            
                            candidate = data['candidates'][0]
                            if 'content' not in candidate:
                                raise ValueError(f"No content in candidate: {candidate}")
                            
                            content = candidate['content']
                            if 'parts' not in content or not content['parts']:
                                raise ValueError(f"No parts in content: {content}")
                            
                            response_text = content['parts'][0].get('text', '')
                            if not response_text:
                                raise ValueError("Empty response text")
                            
                            # Extract token usage
                            tokens_used = None
                            if 'usageMetadata' in data:
                                tokens_used = data['usageMetadata'].get('candidatesTokenCount')
                            
                            return ModelResponse(
                                model_name=f"Gemini ({model})",
                                response=response_text,
                                tokens_used=tokens_used,
                                response_time=time.time() - start_time
                            )
                            
                        except json.JSONDecodeError as e:
                            detailed_error = DetailedError(
                                error_type="json_parse",
                                error_message=f"Failed to parse JSON response: {str(e)}",
                                http_status=response.status,
                                response_headers=response_headers,
                                request_url=url,
                                request_payload=payload,
                                raw_response=response_text[:1000],
                                troubleshooting_tips=[
                                    "🔧 The API returned invalid JSON - this suggests a server error",
                                    "🔧 Check if the model name is correct",
                                    "🔧 Try again in a few minutes",
                                    f"🔧 Raw response preview: {response_text[:200]}..."
                                ]
                            )
                            return ModelResponse(
                                model_name=f"Gemini ({model})",
                                response="",
                                error=f"JSON parsing failed: {str(e)}",
                                detailed_error=detailed_error,
                                response_time=time.time() - start_time
                            )
                        
                        except ValueError as e:
                            detailed_error = DetailedError(
                                error_type="response_structure",
                                error_message=f"Invalid response structure: {str(e)}",
                                http_status=response.status,
                                response_headers=response_headers,
                                request_url=url,
                                request_payload=payload,
                                raw_response=response_text[:1000],
                                troubleshooting_tips=[
                                    "🔧 The API response doesn't match expected structure",
                                    "🔧 This might indicate an API version mismatch",
                                    "🔧 Try a different model or check Gemini API documentation",
                                    f"🔧 Response structure: {response_text[:300]}..."
                                ]
                            )
                            return ModelResponse(
                                model_name=f"Gemini ({model})",
                                response="",
                                error=f"Response structure error: {str(e)}",
                                detailed_error=detailed_error,
                                response_time=time.time() - start_time
                            )
                    
                    else:
                        # Handle HTTP errors with detailed troubleshooting
                        tips = debugger.get_troubleshooting_tips("http_error", response.status)
                        
                        # Try to parse error details from response
                        error_details = ""
                        try:
                            error_data = json.loads(response_text)
                            if 'error' in error_data:
                                error_details = error_data['error'].get('message', '')
                        except:
                            error_details = response_text[:500]
                        
                        detailed_error = DetailedError(
                            error_type="http_error",
                            error_message=f"HTTP {response.status}: {error_details}",
                            http_status=response.status,
                            response_headers=response_headers,
                            request_url=url,
                            request_payload=payload,
                            raw_response=response_text,
                            troubleshooting_tips=tips
                        )
                        
                        return ModelResponse(
                            model_name=f"Gemini ({model})",
                            response="",
                            error=f"HTTP {response.status}: {error_details}",
                            detailed_error=detailed_error,
                            response_time=time.time() - start_time
                        )
        
        except asyncio.TimeoutError:
            detailed_error = DetailedError(
                error_type="timeout",
                error_message="Request timed out",
                troubleshooting_tips=debugger.get_troubleshooting_tips("timeout")
            )
            return ModelResponse(
                model_name=f"Gemini ({model})",
                response="",
                error="Request timed out",
                detailed_error=detailed_error,
                response_time=time.time() - start_time
            )
        
        except aiohttp.ClientError as e:
            detailed_error = DetailedError(
                error_type="connection",
                error_message=f"Connection error: {str(e)}",
                troubleshooting_tips=debugger.get_troubleshooting_tips("connection") + [
                    f"🔧 Specific error: {type(e).__name__}: {str(e)}"
                ]
            )
            return ModelResponse(
                model_name=f"Gemini ({model})",
                response="",
                error=f"Connection error: {str(e)}",
                detailed_error=detailed_error,
                response_time=time.time() - start_time
            )
        
        except Exception as e:
            # Catch-all for unexpected errors with full stack trace
            stack_trace = traceback.format_exc()
            logger.error(f"Unexpected error in Gemini query: {stack_trace}")
            
            detailed_error = DetailedError(
                error_type="unexpected",
                error_message=f"Unexpected error: {str(e)}",
                troubleshooting_tips=[
                    "🔧 An unexpected error occurred",
                    "🔧 Check the application logs for full stack trace",
                    "🔧 This might be a bug - please report it",
                    f"🔧 Error type: {type(e).__name__}",
                    f"🔧 Stack trace: {stack_trace[-500:]}"  # Last 500 chars of stack trace
                ]
            )
            return ModelResponse(
                model_name=f"Gemini ({model})",
                response="",
                error=f"Unexpected error: {str(e)}",
                detailed_error=detailed_error,
                response_time=time.time() - start_time
            )
    
    async def query_single_model(self, prompt: str, provider: str, api_key: str, model_id: str, max_tokens: int = 2000) -> ModelResponse:
        """Query a single model - used for retries"""
        if provider == 'claude':
            return await self.query_claude(prompt, api_key, model_id, max_tokens)
        elif provider == 'chatgpt':
            return await self.query_chatgpt(prompt, api_key, model_id, max_tokens)
        elif provider == 'gemini':
            return await self.query_gemini(prompt, api_key, model_id, max_tokens)
        else:
            return ModelResponse(
                model_name=f"{provider}",
                response="",
                error="Unknown provider",
                response_time=0.0
            )

    async def query_all_models(self, prompt: str, api_keys: Dict[str, str], model_selections: Dict[str, str], max_tokens: int = 2000) -> List[ModelResponse]:
        """Query all models in parallel"""
        tasks = []
        
        if api_keys.get('claude') and model_selections.get('claude'):
            tasks.append(self.query_claude(prompt, api_keys['claude'], model_selections['claude'], max_tokens))
        if api_keys.get('chatgpt') and model_selections.get('chatgpt'):
            tasks.append(self.query_chatgpt(prompt, api_keys['chatgpt'], model_selections['chatgpt'], max_tokens))
        if api_keys.get('gemini') and model_selections.get('gemini'):
            tasks.append(self.query_gemini(prompt, api_keys['gemini'], model_selections['gemini'], max_tokens))
        
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions that occurred
        results = []
        for response in responses:
            if isinstance(response, Exception):
                results.append(ModelResponse(
                    model_name="Unknown",
                    response="",
                    error=str(response)
                ))
            else:
                results.append(response)
        
        return results

def display_detailed_error(response: ModelResponse):
    """Display detailed error information in Streamlit"""
    if not response.detailed_error:
        st.error(f"Error: {response.error}")
        return
    
    error = response.detailed_error
    
    # Main error display
    st.error(f"**{response.model_name}** - {error.error_type.title()} Error")
    st.write(f"**Message:** {error.error_message}")
    
    # HTTP details if available
    if error.http_status:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("HTTP Status", error.http_status)
        with col2:
            st.metric("Response Time", f"{response.response_time:.2f}s")
    
    # Troubleshooting tips
    if error.troubleshooting_tips:
        with st.expander("🔧 Troubleshooting Tips", expanded=True):
            for tip in error.troubleshooting_tips:
                st.write(f"• {tip}")
    
    # Technical details for debugging
    with st.expander("🔍 Technical Details", expanded=False):
        st.write(f"**Error Type:** {error.error_type}")
        st.write(f"**Timestamp:** {error.timestamp}")
        
        if error.request_url:
            st.write(f"**Request URL:** {error.request_url}")
        
        if error.request_payload:
            st.write("**Request Payload:**")
            st.json(error.request_payload)
        
        if error.response_headers:
            st.write("**Response Headers:**")
            st.json(error.response_headers)
        
        if error.raw_response:
            st.write("**Raw Response:**")
            st.code(error.raw_response[:1000] + ("..." if len(error.raw_response) > 1000 else ""))

# Test function for API validation
async def test_gemini_api(api_key: str, model: str = "gemini-1.5-pro") -> Dict[str, Any]:
    """Test Gemini API with detailed diagnostics"""
    agent = MultiModelAgent()
    
    # Simple test prompt
    test_prompt = "Hello, please respond with 'API test successful'"
    
    result = await agent.query_gemini(test_prompt, api_key, model, max_tokens=50)
    
    return {
        "success": not bool(result.error),
        "response": result,
        "diagnostics": {
            "api_key_length": len(api_key),
            "model_used": model,
            "response_time": result.response_time,
            "error_type": result.detailed_error.error_type if result.detailed_error else None
        }
    }

def show_gemini_debug_panel():
    """Show debug panel for Gemini API testing"""
    st.subheader("🔧 Gemini API Debug Panel")
    
    # API key input
    debug_api_key = st.text_input(
        "Gemini API Key for Testing", 
        type="password",
        help="Enter your Gemini API key to run diagnostics"
    )
    
    if debug_api_key:
        # Key validation
        debugger = GeminiDebugger()
        is_valid, issues = debugger.validate_api_key(debug_api_key)
        
        if not is_valid:
            st.error("API Key Issues:")
            for issue in issues:
                st.write(f"• {issue}")
        else:
            st.success("✅ API key format looks valid")
        
        # Model selection for testing
        test_model = st.selectbox(
            "Model to Test",
            ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-1.5-pro-latest", "gemini-pro"],
            help="Select a model to test"
        )
        
        # Test button
        if st.button("🧪 Run API Test"):
            with st.spinner("Testing Gemini API..."):
                test_result = asyncio.run(test_gemini_api(debug_api_key, test_model))
            
            if test_result["success"]:
                st.success("✅ API test successful!")
                st.write(f"Response: {test_result['response'].response}")
                st.json(test_result["diagnostics"])
            else:
                st.error("❌ API test failed")
                display_detailed_error(test_result["response"])

def get_provider_from_model_name(model_name: str) -> str:
    """Extract provider from model response name"""
    if "Claude" in model_name:
        return "claude"
    elif "ChatGPT" in model_name:
        return "chatgpt"
    elif "Gemini" in model_name:
        return "gemini"
    return "unknown"

def display_side_by_side(responses: List[ModelResponse]):
    """Display responses in side-by-side columns with copy buttons and retry"""
    cols = st.columns(len(responses))
    
    for i, response in enumerate(responses):
        with cols[i]:
            if response.error:
                # Use enhanced error display for Gemini
                if "Gemini" in response.model_name and response.detailed_error:
                    display_detailed_error(response)
                else:
                    st.error(f"**{response.model_name}** - Error: {response.error}")
                
                # Add retry functionality for failed responses
                provider = get_provider_from_model_name(response.model_name)
                if provider != "unknown":
                    if st.button(f"🔄 Retry {provider.title()}", key=f"retry_{provider}_{i}"):
                        st.session_state[f"retry_{provider}"] = True
                        st.rerun()
            else:
                # Success header with metrics and copy button
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.success(f"**{response.model_name}** ({response.response_time:.2f}s)")
                with col2:
                    # Native Streamlit copy functionality with feedback
                    if st.button("📋", key=f"copy_sb_{i}", help="Copy response to clipboard"):
                        # Store the response in session state for copying
                        st.session_state[f"copied_text_{i}"] = response.response
                        st.success("✅ Copied!")
                        time.sleep(1)
                        st.rerun()
                
                # Response content in expandable code block for easy copying
                with st.expander("📝 Response", expanded=True):
                    st.text_area(
                        "Response content (select all and copy):",
                        value=response.response,
                        height=200,
                        key=f"response_text_{i}",
                        help="Select all text (Ctrl+A/Cmd+A) and copy (Ctrl+C/Cmd+C)"
                    )
                
                # Token info
                if response.tokens_used:
                    st.caption(f"Tokens: {response.tokens_used}")

def display_sequential(responses: List[ModelResponse]):
    """Display responses one after another with copy buttons and retry"""
    for i, response in enumerate(responses):
        st.subheader(f"{response.model_name}")
        
        if response.error:
            # Use enhanced error display for Gemini
            if "Gemini" in response.model_name and response.detailed_error:
                display_detailed_error(response)
            else:
                st.error(f"Error: {response.error}")
            
            # Add retry functionality for failed responses
            provider = get_provider_from_model_name(response.model_name)
            if provider != "unknown":
                if st.button(f"🔄 Retry {provider.title()}", key=f"retry_seq_{provider}_{i}"):
                    st.session_state[f"retry_{provider}"] = True
                    st.rerun()
        else:
            # Metrics and controls
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Response Time", f"{response.response_time:.2f}s")
            with col2:
                st.metric("Tokens", response.tokens_used or "N/A")
            with col3:
                if st.button(f"👍 Select", key=f"select_{response.model_name}_{i}"):
                    st.session_state[f"selected_response"] = response.response
                    st.success("Response selected!")
            with col4:
                # Copy button with feedback
                if st.button("📋 Copy", key=f"copy_seq_{i}"):
                    st.session_state[f"copied_text_{i}"] = response.response
                    st.success("✅ Copied to clipboard!")
                    time.sleep(1)
                    st.rerun()
            
            # Response content in copyable format
            st.text_area(
                "Response (select all and copy):",
                value=response.response,
                height=150,
                key=f"seq_response_{i}",
                help="Select all (Ctrl+A/Cmd+A) and copy (Ctrl+C/Cmd+C)"
            )
        
        st.divider()

def display_detailed_analysis(responses: List[ModelResponse]):
    """Display with detailed comparison metrics, copy buttons and retry"""
    # Summary metrics
    st.subheader("📊 Comparison Summary")
    
    successful_responses = [r for r in responses if not r.error]
    if successful_responses:
        col1, col2, col3 = st.columns(3)
        with col1:
            avg_time = sum(r.response_time for r in successful_responses) / len(successful_responses)
            st.metric("Avg Response Time", f"{avg_time:.2f}s")
        with col2:
            total_tokens = sum(r.tokens_used or 0 for r in successful_responses)
            st.metric("Total Tokens", total_tokens)
        with col3:
            success_rate = len(successful_responses) / len(responses) * 100
            st.metric("Success Rate", f"{success_rate:.1f}%")
    
    # Individual responses
    st.subheader("📝 Individual Responses")
    
    for i, response in enumerate(responses):
        with st.expander(f"{response.model_name} - {'✅' if not response.error else '❌'}", expanded=True):
            if response.error:
                # Use enhanced error display for Gemini
                if "Gemini" in response.model_name and response.detailed_error:
                    display_detailed_error(response)
                else:
                    st.error(f"Error: {response.error}")
                
                # Add retry functionality for failed responses
                provider = get_provider_from_model_name(response.model_name)
                if provider != "unknown":
                    if st.button(f"🔄 Retry {provider.title()}", key=f"retry_detail_{provider}_{i}"):
                        st.session_state[f"retry_{provider}"] = True
                        st.rerun()
            else:
                # Header with copy button and metrics
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    st.caption(f"Time: {response.response_time:.2f}s | Tokens: {response.tokens_used or 'N/A'}")
                with col2:
                    # Copy button with immediate feedback
                    if st.button("📋 Copy", key=f"copy_detail_{i}"):
                        # Create a temporary code block for easy copying
                        st.session_state[f"show_copy_{i}"] = True
                        st.rerun()
                with col3:
                    if st.session_state.get(f"show_copy_{i}"):
                        if st.button("✅ Done", key=f"hide_copy_{i}"):
                            st.session_state[f"show_copy_{i}"] = False
                            st.rerun()
                
                # Show copyable text area when copy is clicked
                if st.session_state.get(f"show_copy_{i}"):
                    st.info("👇 Select all text below and copy (Ctrl+A then Ctrl+C)")
                    st.code(response.response, language=None)
                
                # Always show the formatted response
                st.markdown("**Response:**")
                st.markdown(response.response)

# Streamlit UI
def main():
    st.set_page_config(
        page_title="Multi-Model AI Comparison", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Load environment variables
    env_keys = load_env_api_keys()
    
    # Initialize session state for persistent data
    if 'api_keys' not in st.session_state:
        # Initialize with environment variables if available
        st.session_state.api_keys = {k: v for k, v in env_keys.items() if v}
    if 'model_selections' not in st.session_state:
        st.session_state.model_selections = {}
    if 'last_responses' not in st.session_state:
        st.session_state.last_responses = None
    if 'last_prompt' not in st.session_state:
        st.session_state.last_prompt = ""
    if 'remember_keys' not in st.session_state:
        st.session_state.remember_keys = bool(any(env_keys.values()))  # Auto-enable if env vars found
    if 'retry_claude' not in st.session_state:
        st.session_state.retry_claude = False
    if 'retry_chatgpt' not in st.session_state:
        st.session_state.retry_chatgpt = False
    if 'retry_gemini' not in st.session_state:
        st.session_state.retry_gemini = False
    if 'show_all_text' not in st.session_state:
        st.session_state.show_all_text = False
    if 'show_statistics' not in st.session_state:
        st.session_state.show_statistics = False
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        # Show info about loaded environment variables
        if any(env_keys.values()):
            loaded_keys = [k for k, v in env_keys.items() if v]
            st.sidebar.success(f"🔑 Loaded from environment: {', '.join(loaded_keys)}")
    
    # Add health check endpoint for Docker
    if st.query_params.get("health") == "check":
        st.write("OK")
        return
    
    st.title("🤖 Multi-Model AI Comparison Tool")
    st.markdown("Compare responses from Claude, ChatGPT, and Gemini in parallel")
    
    # Add enhanced Gemini debug panel in sidebar
    with st.sidebar.expander("🔧 Gemini Debug Panel", expanded=False):
        show_gemini_debug_panel()
    
    # Sidebar for API keys
    st.sidebar.header("🔑 API Configuration")
    
    # Environment variable status
    if any(env_keys.values()):
        with st.sidebar.expander("📋 Environment Variables", expanded=False):
            for service, key in env_keys.items():
                if key:
                    masked_key = f"{key[:8]}...{key[-4:]}" if len(key) > 12 else key[:4] + "..."
                    st.text(f"{service.upper()}: {masked_key}")
                else:
                    st.text(f"{service.upper()}: Not set")
    
    # Remember keys option
    remember_keys = st.sidebar.checkbox(
        "Remember keys this session",
        value=st.session_state.remember_keys,
        help="Keep API keys until browser refresh. Note: Browser refresh will always clear everything for security."
    )
    st.session_state.remember_keys = remember_keys
    
    if not remember_keys:
        st.sidebar.markdown("🔒 Keys cleared after each interaction")
    else:
        st.sidebar.markdown("🔄 Keys stored until page refresh")
    
    # Warning about refresh behavior
    if remember_keys:
        st.sidebar.warning("⚠️ Page refresh will clear all data (including results)")
    
    # Get API keys (from session state if remembering, otherwise use current values)
    if remember_keys:
        # Use session state, but allow override from environment
        current_api_keys = st.session_state.api_keys.copy()
        # Fill in any missing keys from environment
        for k, v in env_keys.items():
            if v and not current_api_keys.get(k):
                current_api_keys[k] = v
    else:
        # Start fresh each time, but pre-fill with environment variables
        current_api_keys = env_keys.copy()
    
    model_selections = st.session_state.model_selections if remember_keys else {}
    
    # API key inputs with help text and model selection (keeping the existing structure)
    with st.sidebar.expander("🤖 Claude (Anthropic)", expanded=True):
        claude_key = st.text_input(
            "Claude API Key", 
            value=current_api_keys.get('claude', ''),
            type="password", 
            help="Get your key at: https://console.anthropic.com (or set ANTHROPIC_API_KEY env var)"
        )
        if claude_key:
            current_api_keys['claude'] = claude_key
            if remember_keys:
                st.session_state.api_keys['claude'] = claude_key
            
            # Get available models dynamically
            agent = MultiModelAgent()
            with st.spinner("🔍 Discovering Claude models..."):
                claude_models, is_dynamic = asyncio.run(agent.get_available_models('claude', claude_key))
            
            if claude_models:
                claude_model_key = st.selectbox(
                    "Claude Model",
                    options=list(claude_models.keys()),
                    index=0,
                    help="Choose which Claude model to use"
                )
                model_selections['claude'] = claude_models[claude_model_key]
                if remember_keys:
                    st.session_state.model_selections['claude'] = model_selections['claude']
                
                # Show source of API key and model info
                key_source = "env var" if env_keys.get('claude') and claude_key == env_keys['claude'] else "manual"
                model_source = "🔄 dynamic" if is_dynamic else "📋 static"
                st.success(f"✅ Claude key configured ({claude_model_key}) [{key_source}] [{model_source}]")
            else:
                st.error("❌ Failed to load Claude models")
    
    with st.sidebar.expander("🧠 ChatGPT (OpenAI)", expanded=True):
        chatgpt_key = st.text_input(
            "OpenAI API Key", 
            value=current_api_keys.get('chatgpt', ''),
            type="password",
            help="Get your key at: https://platform.openai.com (or set OPENAI_API_KEY env var)"
        )
        if chatgpt_key:
            current_api_keys['chatgpt'] = chatgpt_key
            if remember_keys:
                st.session_state.api_keys['chatgpt'] = chatgpt_key
            
            # Get available models dynamically
            agent = MultiModelAgent()
            with st.spinner("🔍 Discovering OpenAI models..."):
                openai_models, is_dynamic = asyncio.run(agent.get_available_models('chatgpt', chatgpt_key))
            
            if openai_models:
                openai_model_key = st.selectbox(
                    "OpenAI Model",
                    options=list(openai_models.keys()),
                    index=0,
                    help="Choose which OpenAI model to use"
                )
                model_selections['chatgpt'] = openai_models[openai_model_key]
                if remember_keys:
                    st.session_state.model_selections['chatgpt'] = model_selections['chatgpt']
                
                # Show source of API key and model info
                key_source = "env var" if env_keys.get('chatgpt') and chatgpt_key == env_keys['chatgpt'] else "manual"
                model_source = "🔄 dynamic" if is_dynamic else "📋 static"
                st.success(f"✅ OpenAI key configured ({openai_model_key}) [{key_source}] [{model_source}]")
            else:
                st.error("❌ Failed to load OpenAI models")
    
    with st.sidebar.expander("✨ Gemini (Google)", expanded=True):
        gemini_key = st.text_input(
            "Gemini API Key", 
            value=current_api_keys.get('gemini', ''),
            type="password",
            help="Get your key at: https://ai.google.dev (or set GOOGLE_API_KEY env var)"
        )
        if gemini_key:
            current_api_keys['gemini'] = gemini_key
            if remember_keys:
                st.session_state.api_keys['gemini'] = gemini_key
            
            # Get available models dynamically
            agent = MultiModelAgent()
            with st.spinner("🔍 Discovering Gemini models..."):
                gemini_models, is_dynamic = asyncio.run(agent.get_available_models('gemini', gemini_key))
            
            if gemini_models:
                gemini_model_key = st.selectbox(
                    "Gemini Model",
                    options=list(gemini_models.keys()),
                    index=0,
                    help="Choose which Gemini model to use"
                )
                model_selections['gemini'] = gemini_models[gemini_model_key]
                if remember_keys:
                    st.session_state.model_selections['gemini'] = model_selections['gemini']
                
                # Show source of API key and model info
                key_source = "env var" if env_keys.get('gemini') and gemini_key == env_keys['gemini'] else "manual"
                model_source = "🔄 dynamic" if is_dynamic else "📋 static"
                st.success(f"✅ Gemini key configured ({gemini_model_key}) [{key_source}] [{model_source}]")
                
                # Show debug info if using fallback
                if not is_dynamic:
                    st.info("ℹ️ Using fallback models. Check logs for model discovery details.")
            else:
                st.error("❌ Failed to load Gemini models - check API key and logs")
    
    # Show configured models
    if current_api_keys:
        configured_count = len([k for k, v in current_api_keys.items() if v])
        st.sidebar.success(f"🎯 {configured_count} model(s) configured")
    else:
        st.sidebar.warning("⚠️ No API keys configured")
    
    # Main interface
    st.header("📝 Enter your prompt")
    prompt = st.text_area(
        "Prompt", 
        value=st.session_state.last_prompt,
        height=150, 
        placeholder="Enter your prompt here...\n\nExample: Write a short story about a robot who discovers emotions."
    )
    
    # Update session state when prompt changes
    if prompt != st.session_state.last_prompt:
        st.session_state.last_prompt = prompt
    
    # Settings
    col1, col2, col3 = st.columns(3)
    with col1:
        task_type = st.selectbox(
            "Task Type", 
            ["Creative Writing", "Code Generation", "Analysis", "General"],
            help="This helps optimize the comparison view"
        )
    with col2:
        comparison_mode = st.selectbox(
            "Comparison Mode", 
            ["Side by Side", "Sequential", "Detailed Analysis"],
            help="How to display the results"
        )
    with col3:
        max_tokens = st.slider(
            "Max Tokens", 
            min_value=100, 
            max_value=4000, 
            value=2000,
            help="Maximum length of responses"
        )
    
    # Handle retry requests (keeping existing retry logic)
    retry_providers = ['claude', 'chatgpt', 'gemini']
    for provider in retry_providers:
        if st.session_state.get(f"retry_{provider}"):
            st.session_state[f"retry_{provider}"] = False  # Reset the flag
            
            # Show retry interface
            st.subheader(f"🔄 Retry {provider.title()}")
            
            # Get current API key
            current_key = current_api_keys.get(provider)
            if not current_key:
                st.error(f"❌ No API key configured for {provider}")
                continue
            
            # Get available models for retry
            agent = MultiModelAgent()
            with st.spinner(f"🔍 Loading {provider} models..."):
                provider_models, _ = asyncio.run(agent.get_available_models(provider, current_key))
            
            if provider_models:
                # Model selection for retry
                retry_col1, retry_col2 = st.columns([3, 1])
                with retry_col1:
                    selected_model_name = st.selectbox(
                        f"Select {provider.title()} Model for Retry",
                        options=list(provider_models.keys()),
                        key=f"retry_model_{provider}"
                    )
                    selected_model_id = provider_models[selected_model_name]
                
                with retry_col2:
                    if st.button(f"🚀 Retry {provider.title()}", key=f"execute_retry_{provider}"):
                        # Execute retry
                        with st.spinner(f"🔄 Retrying {provider}..."):
                            retry_response = asyncio.run(
                                agent.query_single_model(
                                    st.session_state.last_prompt,
                                    provider,
                                    current_key,
                                    selected_model_id,
                                    max_tokens
                                )
                            )
                        
                        # Update the response in session state
                        if st.session_state.last_responses:
                            # Find and replace the response for this provider
                            for i, resp in enumerate(st.session_state.last_responses):
                                if provider in resp.model_name.lower():
                                    st.session_state.last_responses[i] = retry_response
                                    break
                            else:
                                # If not found, append new response
                                st.session_state.last_responses.append(retry_response)
                        
                        st.success(f"✅ {provider.title()} retry completed!")
                        time.sleep(1)
                        st.rerun()
            else:
                st.error(f"❌ Failed to load {provider} models")
    
    # Add example prompts (keeping existing)
    with st.expander("💡 Example Prompts"):
        example_col1, example_col2 = st.columns(2)
        
        with example_col1:
            st.markdown("**Creative Writing:**")
            if st.button("📚 Short Story", help="Generate a creative short story"):
                st.session_state.example_prompt = "Write a 300-word short story about a time traveler who gets stuck in the year 1823 and has to explain modern technology to the locals."
            
            if st.button("🎭 Character Development", help="Create a character"):
                st.session_state.example_prompt = "Create a detailed character profile for a cyberpunk detective in Neo-Tokyo 2090. Include background, motivations, and unique quirks."
        
        with example_col2:
            st.markdown("**Code Generation:**")
            if st.button("💻 Python Function", help="Generate Python code"):
                st.session_state.example_prompt = "Write a Python function that takes a list of dictionaries and returns the top 3 items sorted by a specified key. Include error handling and type hints."
            
            if st.button("🌐 Web Component", help="Generate web code"):
                st.session_state.example_prompt = "Create a responsive React component for a product card that displays an image, title, price, and rating with hover effects."
    
    # Use example prompt if set
    if hasattr(st.session_state, 'example_prompt'):
        prompt = st.session_state.example_prompt
        st.session_state.last_prompt = prompt
        del st.session_state.example_prompt
        st.rerun()
    
    # Query button
    if st.button("🚀 Query All Models", type="primary", use_container_width=True):
        if not prompt.strip():
            st.error("❌ Please enter a prompt")
        elif not current_api_keys or not model_selections:
            st.error("❌ Please enter at least one API key and select a model in the sidebar")
        else:
            agent = MultiModelAgent()
            
            # Show progress and timing
            start_time = time.time()
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                with st.spinner("🔄 Querying models..."):
                    status_text.text("🚀 Sending requests to all models...")
                    progress_bar.progress(0.3)
                    
                    # Run async function in Streamlit
                    responses = asyncio.run(agent.query_all_models(prompt, current_api_keys, model_selections, max_tokens))
                    
                    progress_bar.progress(1.0)
                    total_time = time.time() - start_time
                    status_text.text(f"✅ Completed in {total_time:.2f} seconds")
                    
                    # Store responses in session state for persistence
                    st.session_state.last_responses = responses
                    st.session_state.last_prompt = prompt
                    
                    # Small delay to show completion
                    time.sleep(0.5)
                    progress_bar.empty()
                    status_text.empty()
                    
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                logger.error(f"Error querying models: {e}")
    
    # Display results from session state if available (keeping existing display logic)
    if st.session_state.last_responses:
        responses = st.session_state.last_responses
        
        # Add a separator and show which prompt these results are for
        st.markdown("---")
        st.subheader("📊 Results")
        if st.session_state.last_prompt:
            with st.expander("💬 Prompt Used", expanded=False):
                st.write(st.session_state.last_prompt)
        
        # Display results
        if comparison_mode == "Side by Side":
            display_side_by_side(responses)
        elif comparison_mode == "Sequential":
            display_sequential(responses)
        else:
            display_detailed_analysis(responses)
        
        # Add export options (keeping existing export logic)
        st.markdown("---")
        st.subheader("📤 Export Options")
        export_col1, export_col2, export_col3 = st.columns(3)
        
        with export_col1:
            st.markdown("**📋 Copy All Responses**")
            all_text = f"Prompt: {st.session_state.last_prompt}\n\n"
            for resp in responses:
                if not resp.error:
                    all_text += f"=== {resp.model_name} ===\n{resp.response}\n\n"
            
            # Show text area for easy copying
            if st.button("📋 Show All Responses", key="show_all_responses"):
                st.session_state.show_all_text = True
                st.rerun()
            
            if st.session_state.get("show_all_text"):
                st.text_area("All Responses (select all and copy):", all_text, height=200, key="all_responses_text")
                if st.button("✅ Hide", key="hide_all_responses"):
                    st.session_state.show_all_text = False
                    st.rerun()
        
        with export_col2:
            st.markdown("**📊 Export as JSON**")
            export_data = {
                "prompt": st.session_state.last_prompt,
                "timestamp": datetime.now().isoformat(),
                "responses": [
                    {
                        "model": resp.model_name,
                        "response": resp.response,
                        "response_time": resp.response_time,
                        "tokens_used": resp.tokens_used,
                        "error": resp.error
                    }
                    for resp in responses
                ]
            }
            st.download_button(
                "💾 Download JSON",
                data=json.dumps(export_data, indent=2),
                file_name=f"ai_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                help="Download results as JSON file"
            )
        
        with export_col3:
            st.markdown("**📈 Statistics**")
            if st.button("📈 Show Statistics", key="show_stats"):
                st.session_state.show_statistics = True
                st.rerun()
            
            if st.session_state.get("show_statistics"):
                successful = [r for r in responses if not r.error]
                if successful:
                    stats = {
                        "total_models": len(responses),
                        "successful_responses": len(successful),
                        "average_response_time": f"{sum(r.response_time for r in successful) / len(successful):.2f}s",
                        "total_tokens": sum(r.tokens_used or 0 for r in successful),
                        "fastest_model": min(successful, key=lambda x: x.response_time).model_name,
                        "most_tokens": max(successful, key=lambda x: x.tokens_used or 0).model_name if any(r.tokens_used for r in successful) else "N/A"
                    }
                    st.json(stats)
                    if st.button("✅ Hide Stats", key="hide_stats"):
                        st.session_state.show_statistics = False
                        st.rerun()
        
        # Clear results button
        if st.button("🗑️ Clear Results", help="Clear current results to start fresh"):
            # Clear all result-related session state
            st.session_state.last_responses = None
            st.session_state.last_prompt = ""
            st.session_state.show_all_text = False
            st.session_state.show_statistics = False
            # Clear any copy-related states
            for key in list(st.session_state.keys()):
                if key.startswith('copied_text_') or key.startswith('show_copy_'):
                    del st.session_state[key]
            st.rerun()
    
    # Footer (keeping existing)
    st.markdown("---")
    footer_col1, footer_col2, footer_col3 = st.columns(3)
    
    with footer_col1:
        if remember_keys:
            st.markdown("🔄 **Session**: Keys & results until refresh")
        else:
            st.markdown("🔒 **Security**: Keys cleared after each action")
    with footer_col2:
        st.markdown("💰 **Cost**: Model discovery free, retries cost credits")
    with footer_col3:
        if any(env_keys.values()):
            st.markdown("🔧 **Environment**: Variables loaded")
        else:
            st.markdown("📋 **Copy**: Native Streamlit clipboard support")
    
    # Environment variable instructions (keeping existing)
    with st.expander("🔧 Environment Variable Setup"):
        st.markdown("""
        **To avoid entering API keys manually, set these environment variables:**
        
        ```bash
        export ANTHROPIC_API_KEY="your_claude_key_here"
        export OPENAI_API_KEY="your_openai_key_here" 
        export GOOGLE_API_KEY="your_gemini_key_here"
        ```
        
        **Docker users:**
        ```bash
        # Add to your docker run command:
        docker run -e ANTHROPIC_API_KEY="your_key" \\
                   -e OPENAI_API_KEY="your_key" \\
                   -e GOOGLE_API_KEY="your_key" \\
                   multi-model-ai-comparison
        ```
        
        **Dynamic Model Discovery:**
        - 🔄 **Dynamic**: Models fetched from API (latest available)
        - 📋 **Static**: Fallback list if API unavailable  
        - 🆓 **Free**: Model discovery doesn't cost tokens
        - ⚡ **Cached**: Results cached for 5 minutes
        
        **Retry & Copy Features:**
        - 🔄 **Retry**: Failed requests can be retried with different models
        - 📋 **Copy**: Native text areas for reliable copying (Ctrl+A, Ctrl+C)
        - 🎯 **In-place updates**: Retries update results without re-running all models
        - 🔧 **Model switching**: Change model during retry
        - ✅ **Visual feedback**: Clear copy and selection indicators
        
        **Note**: Environment variables are loaded on startup and can be overridden manually in the UI.
        """)
    
    # Enhanced debug section
    with st.expander("🔧 Debug & Troubleshooting"):
        st.markdown("""
        **Enhanced Gemini Debugging:**
        
        The app now includes enhanced error reporting for Gemini API issues:
        - ✅ **API Key Validation**: Checks key format before making requests
        - 🔍 **Detailed Error Messages**: Specific troubleshooting tips for each error type
        - 📋 **Request/Response Logging**: Full request details logged for debugging
        - 🛠️ **Debug Panel**: Test API keys and diagnose issues in the sidebar
        
        **Common Gemini Issues & Solutions:**
        
        **API Key Problems:**
        - `validation failed`: Key format invalid or wrong service key
        - `401 Unauthorized`: Invalid or expired API key
        - Solution: Get fresh key from https://aistudio.google.com/app/apikey
        
        **Model Availability:**
        - `404 Not Found`: Model doesn't exist or not available in your region
        - `403 Forbidden`: Model access restricted or quota exceeded
        - Solution: Try different model (gemini-1.5-pro vs gemini-1.5-flash)
        
        **Response Structure Issues:**
        - `json_parse`: Invalid JSON response (server error)
        - `response_structure`: Unexpected API response format
        - Solution: Check logs for raw response, try again later
        
        **Testing Commands:**
        ```bash
        # Test your Gemini API key
        curl "https://generativelanguage.googleapis.com/v1beta/models?key=YOUR_KEY"
        
        # Test simple generation
        curl -X POST \\
          "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent?key=YOUR_KEY" \\
          -H "Content-Type: application/json" \\
          -d '{"contents": [{"parts": [{"text": "Hello"}]}]}'
        ```
        
        **Debug Logs:**
        - Check container logs: `make logs` or `docker logs container_name`
        - Look for "🚀 Gemini API Request" and "📥 Gemini API Response" entries
        - Full request/response details are logged for troubleshooting
        
        **Getting Help:**
        - Use the Gemini Debug Panel in the sidebar to test your API key
        - Check the detailed error information in failed responses
        - Review troubleshooting tips provided for each error type
        """)

if __name__ == "__main__":
    main()