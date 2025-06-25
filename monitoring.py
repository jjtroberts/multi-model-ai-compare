# monitoring.py - Enhanced monitoring and observability
import asyncio
import time
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import threading
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

@dataclass
class APIMetrics:
    """Metrics for API performance tracking"""
    provider: str
    model: str
    request_count: int = 0
    success_count: int = 0
    error_count: int = 0
    total_response_time: float = 0.0
    total_tokens: int = 0
    last_success: Optional[datetime] = None
    last_error: Optional[datetime] = None
    error_types: Dict[str, int] = None
    
    def __post_init__(self):
        if self.error_types is None:
            self.error_types = defaultdict(int)
    
    @property
    def success_rate(self) -> float:
        return (self.success_count / self.request_count) if self.request_count > 0 else 0.0
    
    @property
    def avg_response_time(self) -> float:
        return (self.total_response_time / self.success_count) if self.success_count > 0 else 0.0
    
    @property
    def avg_tokens_per_request(self) -> float:
        return (self.total_tokens / self.success_count) if self.success_count > 0 else 0.0

class MetricsCollector:
    """Collect and track API performance metrics"""
    
    def __init__(self, retention_hours: int = 24):
        self.metrics: Dict[str, APIMetrics] = {}
        self.request_history: deque = deque(maxlen=1000)  # Keep last 1000 requests
        self.retention_hours = retention_hours
        self.lock = threading.Lock()
    
    def get_key(self, provider: str, model: str) -> str:
        return f"{provider}:{model}"
    
    def record_request(self, provider: str, model: str, success: bool, 
                      response_time: float, tokens: int = 0, error_type: str = None):
        """Record API request metrics"""
        with self.lock:
            key = self.get_key(provider, model)
            
            if key not in self.metrics:
                self.metrics[key] = APIMetrics(provider=provider, model=model)
            
            metric = self.metrics[key]
            metric.request_count += 1
            
            # Record in history
            self.request_history.append({
                'timestamp': datetime.now(),
                'provider': provider,
                'model': model,
                'success': success,
                'response_time': response_time,
                'tokens': tokens,
                'error_type': error_type
            })
            
            if success:
                metric.success_count += 1
                metric.total_response_time += response_time
                metric.total_tokens += tokens
                metric.last_success = datetime.now()
            else:
                metric.error_count += 1
                metric.last_error = datetime.now()
                if error_type:
                    metric.error_types[error_type] += 1
    
    def get_metrics(self, provider: str = None, model: str = None) -> Dict[str, APIMetrics]:
        """Get metrics, optionally filtered by provider/model"""
        with self.lock:
            if provider and model:
                key = self.get_key(provider, model)
                return {key: self.metrics.get(key)} if key in self.metrics else {}
            elif provider:
                return {k: v for k, v in self.metrics.items() if v.provider == provider}
            else:
                return dict(self.metrics)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get overall summary metrics"""
        with self.lock:
            total_requests = sum(m.request_count for m in self.metrics.values())
            total_successes = sum(m.success_count for m in self.metrics.values())
            total_errors = sum(m.error_count for m in self.metrics.values())
            
            # Recent performance (last hour)
            one_hour_ago = datetime.now() - timedelta(hours=1)
            recent_requests = [r for r in self.request_history if r['timestamp'] > one_hour_ago]
            
            return {
                'total_requests': total_requests,
                'total_successes': total_successes,
                'total_errors': total_errors,
                'overall_success_rate': total_successes / total_requests if total_requests > 0 else 0.0,
                'recent_requests_1h': len(recent_requests),
                'recent_success_rate_1h': sum(1 for r in recent_requests if r['success']) / len(recent_requests) if recent_requests else 0.0,
                'providers_active': len(set(m.provider for m in self.metrics.values())),
                'models_active': len(self.metrics)
            }
    
    def cleanup_old_data(self):
        """Remove old data beyond retention period"""
        cutoff = datetime.now() - timedelta(hours=self.retention_hours)
        
        with self.lock:
            # Clean request history
            while self.request_history and self.request_history[0]['timestamp'] < cutoff:
                self.request_history.popleft()

# Global metrics collector instance
metrics_collector = MetricsCollector()

@asynccontextmanager
async def track_api_call(provider: str, model: str):
    """Context manager to track API call metrics"""
    start_time = time.time()
    success = False
    tokens = 0
    error_type = None
    
    try:
        yield
        success = True
    except Exception as e:
        error_type = type(e).__name__
        raise
    finally:
        response_time = time.time() - start_time
        metrics_collector.record_request(
            provider=provider,
            model=model,
            success=success,
            response_time=response_time,
            tokens=tokens,
            error_type=error_type
        )

# health_check.py - API health monitoring
class APIHealthChecker:
    """Monitor API health and availability"""
    
    def __init__(self):
        self.health_status = {}
        self.check_interval = 300  # 5 minutes
        self.running = False
    
    async def check_api_health(self, provider: str, api_key: str) -> Dict[str, Any]:
        """Check health of a specific API"""
        start_time = time.time()
        
        try:
            if provider == 'gemini':
                return await self._check_gemini_health(api_key)
            elif provider == 'claude':
                return await self._check_claude_health(api_key)
            elif provider == 'chatgpt':
                return await self._check_openai_health(api_key)
            else:
                return {
                    'provider': provider,
                    'healthy': False,
                    'error': 'Unknown provider',
                    'response_time': time.time() - start_time
                }
        except Exception as e:
            return {
                'provider': provider,
                'healthy': False,
                'error': str(e),
                'response_time': time.time() - start_time
            }
    
    async def _check_gemini_health(self, api_key: str) -> Dict[str, Any]:
        """Check Gemini API health"""
        import aiohttp
        
        url = f'https://generativelanguage.googleapis.com/v1beta/models?key={api_key}'
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    model_count = len(data.get('models', []))
                    return {
                        'provider': 'gemini',
                        'healthy': True,
                        'model_count': model_count,
                        'response_time': time.time()
                    }
                else:
                    return {
                        'provider': 'gemini',
                        'healthy': False,
                        'error': f'HTTP {response.status}',
                        'response_time': time.time()
                    }
    
    async def _check_claude_health(self, api_key: str) -> Dict[str, Any]:
        """Check Claude API health"""
        import aiohttp
        
        headers = {
            'x-api-key': api_key,
            'anthropic-version': '2023-06-01'
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get('https://api.anthropic.com/v1/models', headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    model_count = len(data.get('data', []))
                    return {
                        'provider': 'claude',
                        'healthy': True,
                        'model_count': model_count,
                        'response_time': time.time()
                    }
                else:
                    return {
                        'provider': 'claude',
                        'healthy': False,
                        'error': f'HTTP {response.status}',
                        'response_time': time.time()
                    }
    
    async def _check_openai_health(self, api_key: str) -> Dict[str, Any]:
        """Check OpenAI API health"""
        import aiohttp
        
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get('https://api.openai.com/v1/models', headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    model_count = len(data.get('data', []))
                    return {
                        'provider': 'openai',
                        'healthy': True,
                        'model_count': model_count,
                        'response_time': time.time()
                    }
                else:
                    return {
                        'provider': 'openai',
                        'healthy': False,
                        'error': f'HTTP {response.status}',
                        'response_time': time.time()
                    }

# testing.py - Automated testing framework
import pytest
import asyncio
from unittest.mock import Mock, patch

class APITestSuite:
    """Comprehensive testing suite for API integrations"""
    
    def __init__(self):
        self.test_prompts = [
            "Hello, please respond with exactly: 'Test successful'",
            "What is 2 + 2?",
            "Write one sentence about artificial intelligence.",
        ]
    
    async def run_connectivity_tests(self, api_keys: Dict[str, str]) -> Dict[str, Dict]:
        """Test basic connectivity to all APIs"""
        results = {}
        
        for provider, api_key in api_keys.items():
            if not api_key:
                continue
                
            results[provider] = await self._test_provider_connectivity(provider, api_key)
        
        return results
    
    async def _test_provider_connectivity(self, provider: str, api_key: str) -> Dict:
        """Test connectivity to a specific provider"""
        from app import MultiModelAgent  # Import your main agent
        
        agent = MultiModelAgent()
        test_results = {
            'provider': provider,
            'tests_passed': 0,
            'tests_failed': 0,
            'errors': [],
            'response_times': []
        }
        
        # Get available models
        try:
            models, is_dynamic = await agent.get_available_models(provider, api_key)
            if not models:
                test_results['errors'].append('No models available')
                test_results['tests_failed'] += 1
                return test_results
            
            # Test with first available model
            model_id = list(models.values())[0]
            
            # Run connectivity test
            start_time = time.time()
            response = await agent.query_single_model(
                "Hello", provider, api_key, model_id, max_tokens=10
            )
            response_time = time.time() - start_time
            
            if response.error:
                test_results['errors'].append(response.error)
                test_results['tests_failed'] += 1
            else:
                test_results['tests_passed'] += 1
                test_results['response_times'].append(response_time)
                
        except Exception as e:
            test_results['errors'].append(str(e))
            test_results['tests_failed'] += 1
        
        return test_results
    
    async def run_load_test(self, provider: str, api_key: str, model_id: str, 
                           concurrent_requests: int = 3, iterations: int = 5) -> Dict:
        """Run load test against an API"""
        from app import MultiModelAgent
        
        agent = MultiModelAgent()
        results = {
            'provider': provider,
            'concurrent_requests': concurrent_requests,
            'iterations': iterations,
            'total_requests': concurrent_requests * iterations,
            'successful_requests': 0,
            'failed_requests': 0,
            'response_times': [],
            'errors': []
        }
        
        for iteration in range(iterations):
            tasks = []
            for i in range(concurrent_requests):
                task = agent.query_single_model(
                    f"Test request {iteration}-{i}", 
                    provider, api_key, model_id, max_tokens=50
                )
                tasks.append(task)
            
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            for response in responses:
                if isinstance(response, Exception):
                    results['failed_requests'] += 1
                    results['errors'].append(str(response))
                elif response.error:
                    results['failed_requests'] += 1
                    results['errors'].append(response.error)
                else:
                    results['successful_requests'] += 1
                    results['response_times'].append(response.response_time)
        
        if results['response_times']:
            results['avg_response_time'] = sum(results['response_times']) / len(results['response_times'])
            results['max_response_time'] = max(results['response_times'])
            results['min_response_time'] = min(results['response_times'])
        
        return results

# configuration.py - Enhanced configuration management
import os
from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class APIConfig:
    """Configuration for API providers"""
    base_url: str
    timeout: int = 60
    max_retries: int = 3
    retry_delay: float = 1.0
    rate_limit_requests: int = 100
    rate_limit_window: int = 60  # seconds
    
@dataclass
class AppConfig:
    """Application configuration"""
    # API configurations
    claude_config: APIConfig
    openai_config: APIConfig
    gemini_config: APIConfig
    
    # Application settings
    debug_mode: bool = False
    log_level: str = "INFO"
    metrics_enabled: bool = True
    health_check_interval: int = 300
    
    # Security settings
    api_key_rotation_hours: int = 24
    max_session_duration: int = 3600
    
    @classmethod
    def from_environment(cls):
        """Create configuration from environment variables"""
        return cls(
            claude_config=APIConfig(
                base_url=os.getenv('CLAUDE_BASE_URL', 'https://api.anthropic.com/v1'),
                timeout=int(os.getenv('CLAUDE_TIMEOUT', '60')),
                max_retries=int(os.getenv('CLAUDE_MAX_RETRIES', '3'))
            ),
            openai_config=APIConfig(
                base_url=os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1'),
                timeout=int(os.getenv('OPENAI_TIMEOUT', '60')),
                max_retries=int(os.getenv('OPENAI_MAX_RETRIES', '3'))
            ),
            gemini_config=APIConfig(
                base_url=os.getenv('GEMINI_BASE_URL', 'https://generativelanguage.googleapis.com/v1beta'),
                timeout=int(os.getenv('GEMINI_TIMEOUT', '60')),
                max_retries=int(os.getenv('GEMINI_MAX_RETRIES', '3'))
            ),
            debug_mode=os.getenv('DEBUG_MODE', 'false').lower() == 'true',
            log_level=os.getenv('LOG_LEVEL', 'INFO'),
            metrics_enabled=os.getenv('METRICS_ENABLED', 'true').lower() == 'true'
        )

# circuit_breaker.py - Circuit breaker for resilience
from enum import Enum
import time
from typing import Callable, Any

class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class CircuitBreaker:
    """Circuit breaker pattern for API resilience"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
    
    def _on_success(self):
        """Handle successful call"""
        self.failure_count = 0
        self.state = CircuitState.CLOSED
    
    def _on_failure(self):
        """Handle failed call"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
    
    def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset the circuit breaker"""
        return (
            self.last_failure_time and 
            time.time() - self.last_failure_time >= self.recovery_timeout
        )

# Enhanced retry logic with exponential backoff
import random

class RetryStrategy:
    """Enhanced retry strategy with exponential backoff"""
    
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0, 
                 max_delay: float = 60.0, jitter: bool = True):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.jitter = jitter
    
    async def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with retry logic"""
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                
                if attempt == self.max_retries:
                    raise
                
                delay = self._calculate_delay(attempt)
                logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.2f}s")
                await asyncio.sleep(delay)
        
        raise last_exception
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for retry attempt"""
        delay = self.base_delay * (2 ** attempt)
        delay = min(delay, self.max_delay)
        
        if self.jitter:
            delay = delay * (0.5 + random.random() * 0.5)
        
        return delay

# Performance profiler
class PerformanceProfiler:
    """Profile performance of API calls"""
    
    def __init__(self):
        self.profiles = {}
    
    async def profile(self, name: str, func: Callable, *args, **kwargs):
        """Profile a function call"""
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        try:
            result = await func(*args, **kwargs)
            success = True
            error = None
        except Exception as e:
            result = None
            success = False
            error = str(e)
            raise
        finally:
            end_time = time.time()
            end_memory = self._get_memory_usage()
            
            profile_data = {
                'duration': end_time - start_time,
                'memory_delta': end_memory - start_memory,
                'success': success,
                'error': error,
                'timestamp': datetime.now().isoformat()
            }
            
            if name not in self.profiles:
                self.profiles[name] = []
            
            self.profiles[name].append(profile_data)
            
            # Keep only last 100 profiles per function
            if len(self.profiles[name]) > 100:
                self.profiles[name] = self.profiles[name][-100:]
        
        return result
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0
    
    def get_stats(self, name: str) -> Dict[str, Any]:
        """Get performance statistics for a function"""
        if name not in self.profiles:
            return {}
        
        profiles = self.profiles[name]
        successful_profiles = [p for p in profiles if p['success']]
        
        if not successful_profiles:
            return {'error': 'No successful calls'}
        
        durations = [p['duration'] for p in successful_profiles]
        
        return {
            'total_calls': len(profiles),
            'successful_calls': len(successful_profiles),
            'success_rate': len(successful_profiles) / len(profiles),
            'avg_duration': sum(durations) / len(durations),
            'min_duration': min(durations),
            'max_duration': max(durations),
            'total_duration': sum(durations)
        }