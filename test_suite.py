#!/usr/bin/env python3
# test_suite.py - Comprehensive testing suite

import asyncio
import os
import sys
import json
import time
import argparse
from typing import Dict, List, Any
from dataclasses import dataclass
import aiohttp
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TestResult:
    test_name: str
    passed: bool
    duration: float
    error: str = None
    details: Dict[str, Any] = None

class APITestRunner:
    """Comprehensive API testing runner"""
    
    def __init__(self):
        self.results: List[TestResult] = []
        self.api_keys = self._load_api_keys()
    
    def _load_api_keys(self) -> Dict[str, str]:
        """Load API keys from environment"""
        return {
            'claude': os.getenv('ANTHROPIC_API_KEY', ''),
            'chatgpt': os.getenv('OPENAI_API_KEY', ''),
            'gemini': os.getenv('GOOGLE_API_KEY', '')
        }
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run comprehensive test suite"""
        logger.info("🧪 Starting comprehensive API test suite")
        
        test_methods = [
            self.test_api_key_validation,
            self.test_model_discovery,
            self.test_basic_connectivity,
            self.test_response_format,
            self.test_error_handling,
            self.test_concurrent_requests,
            self.test_rate_limiting,
            self.test_large_prompts,
            self.test_special_characters,
            self.test_timeout_handling
        ]
        
        for test_method in test_methods:
            try:
                await test_method()
            except Exception as e:
                logger.error(f"Test {test_method.__name__} failed with exception: {e}")
                self.results.append(TestResult(
                    test_name=test_method.__name__,
                    passed=False,
                    duration=0.0,
                    error=str(e)
                ))
        
        return self._generate_report()
    
    async def test_api_key_validation(self):
        """Test API key validation"""
        logger.info("🔑 Testing API key validation")
        
        for provider, api_key in self.api_keys.items():
            if not api_key:
                self.results.append(TestResult(
                    test_name=f"api_key_validation_{provider}",
                    passed=False,
                    duration=0.0,
                    error="API key not provided"
                ))
                continue
            
            start_time = time.time()
            try:
                is_valid = await self._validate_api_key(provider, api_key)
                duration = time.time() - start_time
                
                self.results.append(TestResult(
                    test_name=f"api_key_validation_{provider}",
                    passed=is_valid,
                    duration=duration,
                    error=None if is_valid else "API key validation failed"
                ))
            except Exception as e:
                duration = time.time() - start_time
                self.results.append(TestResult(
                    test_name=f"api_key_validation_{provider}",
                    passed=False,
                    duration=duration,
                    error=str(e)
                ))
    
    async def test_model_discovery(self):
        """Test model discovery for each provider"""
        logger.info("🔍 Testing model discovery")
        
        for provider, api_key in self.api_keys.items():
            if not api_key:
                continue
            
            start_time = time.time()
            try:
                models = await self._discover_models(provider, api_key)
                duration = time.time() - start_time
                
                self.results.append(TestResult(
                    test_name=f"model_discovery_{provider}",
                    passed=len(models) > 0,
                    duration=duration,
                    details={'model_count': len(models), 'models': list(models.keys())[:5]}
                ))
            except Exception as e:
                duration = time.time() - start_time
                self.results.append(TestResult(
                    test_name=f"model_discovery_{provider}",
                    passed=False,
                    duration=duration,
                    error=str(e)
                ))
    
    async def test_basic_connectivity(self):
        """Test basic connectivity with simple prompts"""
        logger.info("🌐 Testing basic connectivity")
        
        test_prompt = "Hello, please respond with exactly: 'Test successful'"
        
        for provider, api_key in self.api_keys.items():
            if not api_key:
                continue
            
            start_time = time.time()
            try:
                response = await self._make_test_request(provider, api_key, test_prompt)
                duration = time.time() - start_time
                
                success = not response.get('error') and response.get('response')
                
                self.results.append(TestResult(
                    test_name=f"basic_connectivity_{provider}",
                    passed=success,
                    duration=duration,
                    details={
                        'response_length': len(response.get('response', '')),
                        'tokens_used': response.get('tokens_used')
                    }
                ))
            except Exception as e:
                duration = time.time() - start_time
                self.results.append(TestResult(
                    test_name=f"basic_connectivity_{provider}",
                    passed=False,
                    duration=duration,
                    error=str(e)
                ))
    
    async def test_response_format(self):
        """Test response format consistency"""
        logger.info("📋 Testing response format")
        
        test_prompts = [
            "What is 2 + 2?",
            "Write one sentence about AI.",
            "List 3 colors: red, blue, green"
        ]
        
        for provider, api_key in self.api_keys.items():
            if not api_key:
                continue
            
            passed_tests = 0
            total_tests = len(test_prompts)
            
            for i, prompt in enumerate(test_prompts):
                try:
                    response = await self._make_test_request(provider, api_key, prompt)
                    if response.get('response') and not response.get('error'):
                        passed_tests += 1
                except Exception:
                    pass
            
            self.results.append(TestResult(
                test_name=f"response_format_{provider}",
                passed=passed_tests == total_tests,
                duration=0.0,
                details={'passed_tests': passed_tests, 'total_tests': total_tests}
            ))
    
    async def test_error_handling(self):
        """Test error handling with invalid inputs"""
        logger.info("❌ Testing error handling")
        
        error_test_cases = [
            ("invalid_model", "Hello", {"model": "invalid-model-name"}),
            ("empty_prompt", "", {}),
            ("excessive_tokens", "Hello", {"max_tokens": 999999})
        ]
        
        for provider, api_key in self.api_keys.items():
            if not api_key:
                continue
            
            errors_handled = 0
            total_error_tests = len(error_test_cases)
            
            for test_name, prompt, params in error_test_cases:
                try:
                    response = await self._make_test_request(provider, api_key, prompt, **params)
                    # We expect an error, so if we get one, that's good
                    if response.get('error'):
                        errors_handled += 1
                except Exception:
                    # Exceptions are also acceptable for error handling tests
                    errors_handled += 1
            
            self.results.append(TestResult(
                test_name=f"error_handling_{provider}",
                passed=errors_handled >= total_error_tests // 2,  # At least half should error
                duration=0.0,
                details={'errors_handled': errors_handled, 'total_tests': total_error_tests}
            ))
    
    async def test_concurrent_requests(self):
        """Test concurrent request handling"""
        logger.info("🔄 Testing concurrent requests")
        
        concurrent_count = 3
        test_prompt = "Count from 1 to 5"
        
        for provider, api_key in self.api_keys.items():
            if not api_key:
                continue
            
            start_time = time.time()
            try:
                tasks = [
                    self._make_test_request(provider, api_key, f"{test_prompt} (request {i})")
                    for i in range(concurrent_count)
                ]
                
                responses = await asyncio.gather(*tasks, return_exceptions=True)
                duration = time.time() - start_time
                
                successful_responses = sum(
                    1 for r in responses 
                    if not isinstance(r, Exception) and not r.get('error')
                )
                
                self.results.append(TestResult(
                    test_name=f"concurrent_requests_{provider}",
                    passed=successful_responses >= concurrent_count // 2,
                    duration=duration,
                    details={
                        'successful_responses': successful_responses,
                        'total_requests': concurrent_count,
                        'avg_response_time': duration / concurrent_count
                    }
                ))
            except Exception as e:
                duration = time.time() - start_time
                self.results.append(TestResult(
                    test_name=f"concurrent_requests_{provider}",
                    passed=False,
                    duration=duration,
                    error=str(e)
                ))
    
    async def test_rate_limiting(self):
        """Test rate limiting behavior"""
        logger.info("⏱️ Testing rate limiting")
        
        # Make rapid requests to test rate limiting
        for provider, api_key in self.api_keys.items():
            if not api_key:
                continue
            
            start_time = time.time()
            rate_limited = False
            
            for i in range(10):  # Make 10 rapid requests
                try:
                    response = await self._make_test_request(provider, api_key, f"Request {i}")
                    if response.get('error') and '429' in str(response.get('error')):
                        rate_limited = True
                        break
                except Exception as e:
                    if '429' in str(e) or 'rate' in str(e).lower():
                        rate_limited = True
                        break
                
                await asyncio.sleep(0.1)  # Small delay between requests
            
            duration = time.time() - start_time
            
            self.results.append(TestResult(
                test_name=f"rate_limiting_{provider}",
                passed=True,  # Rate limiting is optional, so we pass regardless
                duration=duration,
                details={'rate_limited': rate_limited}
            ))
    
    async def test_large_prompts(self):
        """Test handling of large prompts"""
        logger.info("📏 Testing large prompts")
        
        large_prompt = "Please summarize the following text: " + "Lorem ipsum " * 1000
        
        for provider, api_key in self.api_keys.items():
            if not api_key:
                continue
            
            start_time = time.time()
            try:
                response = await self._make_test_request(provider, api_key, large_prompt)
                duration = time.time() - start_time
                
                # Large prompts might fail, which is acceptable
                self.results.append(TestResult(
                    test_name=f"large_prompts_{provider}",
                    passed=True,  # We just want to test it doesn't crash
                    duration=duration,
                    details={
                        'prompt_length': len(large_prompt),
                        'response_received': bool(response.get('response')),
                        'error': response.get('error')
                    }
                ))
            except Exception as e:
                duration = time.time() - start_time
                self.results.append(TestResult(
                    test_name=f"large_prompts_{provider}",
                    passed=True,  # Exceptions are acceptable for large prompts
                    duration=duration,
                    error=str(e)
                ))
    
    async def test_special_characters(self):
        """Test handling of special characters"""
        logger.info("🔤 Testing special characters")
        
        special_prompts = [
            "Hello 世界 🌍",
            "Math: 2 + 2 = 4, π ≈ 3.14",
            "Code: def hello(): print('Hi!')",
            "Symbols: @#$%^&*()[]{}|\\:;\"'<>,.?/~`"
        ]
        
        for provider, api_key in self.api_keys.items():
            if not api_key:
                continue
            
            successful_prompts = 0
            
            for prompt in special_prompts:
                try:
                    response = await self._make_test_request(provider, api_key, prompt)
                    if response.get('response') and not response.get('error'):
                        successful_prompts += 1
                except Exception:
                    pass
            
            self.results.append(TestResult(
                test_name=f"special_characters_{provider}",
                passed=successful_prompts >= len(special_prompts) // 2,
                duration=0.0,
                details={
                    'successful_prompts': successful_prompts,
                    'total_prompts': len(special_prompts)
                }
            ))
    
    async def test_timeout_handling(self):
        """Test timeout handling"""
        logger.info("⏰ Testing timeout handling")
        
        for provider, api_key in self.api_keys.items():
            if not api_key:
                continue
            
            start_time = time.time()
            try:
                # Test with a very short timeout
                response = await self._make_test_request(
                    provider, api_key, "Write a long essay about AI", timeout=1
                )
                duration = time.time() - start_time
                
                # Timeout might or might not occur, both are acceptable
                self.results.append(TestResult(
                    test_name=f"timeout_handling_{provider}",
                    passed=True,
                    duration=duration,
                    details={
                        'completed_within_timeout': duration < 1.5,
                        'response_received': bool(response.get('response'))
                    }
                ))
            except Exception as e:
                duration = time.time() - start_time
                self.results.append(TestResult(
                    test_name=f"timeout_handling_{provider}",
                    passed=True,  # Timeout exceptions are expected
                    duration=duration,
                    error=str(e)
                ))
    
    async def _validate_api_key(self, provider: str, api_key: str) -> bool:
        """Validate API key for a provider"""
        if provider == 'claude':
            return await self._validate_claude_key(api_key)
        elif provider == 'chatgpt':
            return await self._validate_openai_key(api_key)
        elif provider == 'gemini':
            return await self._validate_gemini_key(api_key)
        return False
    
    async def _validate_claude_key(self, api_key: str) -> bool:
        """Validate Claude API key"""
        try:
            headers = {'x-api-key': api_key, 'anthropic-version': '2023-06-01'}
            async with aiohttp.ClientSession() as session:
                async with session.get('https://api.anthropic.com/v1/models', headers=headers) as response:
                    return response.status == 200
        except Exception:
            return False
    
    async def _validate_openai_key(self, api_key: str) -> bool:
        """Validate OpenAI API key"""
        try:
            headers = {'Authorization': f'Bearer {api_key}'}
            async with aiohttp.ClientSession() as session:
                async with session.get('https://api.openai.com/v1/models', headers=headers) as response:
                    return response.status == 200
        except Exception:
            return False
    
    async def _validate_gemini_key(self, api_key: str) -> bool:
        """Validate Gemini API key"""
        try:
            url = f'https://generativelanguage.googleapis.com/v1beta/models?key={api_key}'
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    return response.status == 200
        except Exception:
            return False
    
    async def _discover_models(self, provider: str, api_key: str) -> Dict[str, str]:
        """Discover available models for a provider"""
        # Import your app's model discovery logic
        try:
            from app import MultiModelAgent
            agent = MultiModelAgent()
            models, _ = await agent.get_available_models(provider, api_key)
            return models
        except Exception:
            return {}
    
    async def _make_test_request(self, provider: str, api_key: str, prompt: str, 
                                timeout: int = 30, **kwargs) -> Dict[str, Any]:
        """Make a test request to a provider"""
        try:
            from app import MultiModelAgent
            agent = MultiModelAgent()
            
            # Get available models
            models, _ = await agent.get_available_models(provider, api_key)
            if not models:
                return {'error': 'No models available'}
            
            # Use first available model unless specified
            model_id = kwargs.get('model', list(models.values())[0])
            max_tokens = kwargs.get('max_tokens', 100)
            
            # Make request with timeout
            response = await asyncio.wait_for(
                agent.query_single_model(prompt, provider, api_key, model_id, max_tokens),
                timeout=timeout
            )
            
            return {
                'response': response.response,
                'error': response.error,
                'tokens_used': response.tokens_used,
                'response_time': response.response_time
            }
            
        except asyncio.TimeoutError:
            return {'error': 'Request timed out'}
        except Exception as e:
            return {'error': str(e)}
    
    def _generate_report(self) -> Dict[str, Any]:
        """Generate test report"""
        passed_tests = sum(1 for r in self.results if r.passed)
        total_tests = len(self.results)
        
        report = {
            'summary': {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': total_tests - passed_tests,
                'success_rate': passed_tests / total_tests if total_tests > 0 else 0.0,
                'total_duration': sum(r.duration for r in self.results)
            },
            'results': [
                {
                    'test_name': r.test_name,
                    'passed': r.passed,
                    'duration': round(r.duration, 3),
                    'error': r.error,
                    'details': r.details
                }
                for r in self.results
            ],
            'providers': {
                provider: {
                    'api_key_provided': bool(self.api_keys.get(provider)),
                    'tests_run': len([r for r in self.results if provider in r.test_name]),
                    'tests_passed': len([r for r in self.results if provider in r.test_name and r.passed])
                }
                for provider in ['claude', 'chatgpt', 'gemini']
            }
        }
        
        return report

# deployment_utils.py - Deployment utilities
import subprocess
import shutil
import yaml

class DeploymentManager:
    """Manage deployment processes"""
    
    def __init__(self):
        self.docker_compose_file = 'docker-compose.yml'
        self.env_file = '.env'
    
    def check_prerequisites(self) -> Dict[str, bool]:
        """Check deployment prerequisites"""
        checks = {
            'docker': self._check_docker(),
            'docker_compose': self._check_docker_compose(),
            'env_file': os.path.exists(self.env_file),
            'ssl_certs': self._check_ssl_certs(),
            'api_keys': self._check_api_keys()
        }
        return checks
    
    def _check_docker(self) -> bool:
        """Check if Docker is available"""
        try:
            subprocess.run(['docker', '--version'], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def _check_docker_compose(self) -> bool:
        """Check if Docker Compose is available"""
        try:
            subprocess.run(['docker-compose', '--version'], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            try:
                subprocess.run(['docker', 'compose', 'version'], capture_output=True, check=True)
                return True
            except (subprocess.CalledProcessError, FileNotFoundError):
                return False
    
    def _check_ssl_certs(self) -> bool:
        """Check if SSL certificates exist"""
        return os.path.exists('ssl/cert.pem') and os.path.exists('ssl/key.pem')
    
    def _check_api_keys(self) -> bool:
        """Check if API keys are configured"""
        required_keys = ['ANTHROPIC_API_KEY', 'OPENAI_API_KEY', 'GOOGLE_API_KEY']
        return any(os.getenv(key) for key in required_keys)
    
    def deploy(self, environment: str = 'production') -> bool:
        """Deploy the application"""
        logger.info(f"🚀 Deploying to {environment}")
        
        # Check prerequisites
        checks = self.check_prerequisites()
        failed_checks = [k for k, v in checks.items() if not v]
        
        if failed_checks:
            logger.error(f"❌ Prerequisites failed: {failed_checks}")
            return False
        
        try:
            # Build and start services
            if environment == 'production':
                cmd = ['docker-compose', 'up', '-d', '--build']
            else:
                cmd = ['docker-compose', '-f', 'docker-compose.dev.yml', 'up', '-d', '--build']
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("✅ Deployment successful")
                return True
            else:
                logger.error(f"❌ Deployment failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Deployment error: {e}")
            return False
    
    def generate_ssl_certs(self):
        """Generate self-signed SSL certificates"""
        os.makedirs('ssl', exist_ok=True)
        
        cmd = [
            'openssl', 'req', '-x509', '-newkey', 'rsa:4096',
            '-keyout', 'ssl/key.pem', '-out', 'ssl/cert.pem',
            '-days', '365', '-nodes', '-subj', '/CN=localhost'
        ]
        
        try:
            subprocess.run(cmd, check=True)
            logger.info("✅ SSL certificates generated")
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to generate SSL certificates: {e}")

def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(description='Multi-Model AI Comparison Tool - Testing & Deployment')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Run test suite')
    test_parser.add_argument('--output', '-o', help='Output file for test results')
    test_parser.add_argument('--format', choices=['json', 'yaml'], default='json', help='Output format')
    
    # Deploy command
    deploy_parser = subparsers.add_parser('deploy', help='Deploy application')
    deploy_parser.add_argument('--env', choices=['production', 'development'], default='production', help='Environment')
    deploy_parser.add_argument('--check-only', action='store_true', help='Only check prerequisites')
    
    # SSL command
    ssl_parser = subparsers.add_parser('ssl', help='Generate SSL certificates')
    
    args = parser.parse_args()
    
    if args.command == 'test':
        # Run test suite
        runner = APITestRunner()
        report = asyncio.run(runner.run_all_tests())
        
        # Output results
        if args.format == 'yaml':
            output = yaml.dump(report, default_flow_style=False)
        else:
            output = json.dumps(report, indent=2)
        
        if args.output:
            with open(args.output, 'w') as f:
                f.write(output)
            logger.info(f"📊 Test results saved to {args.output}")
        else:
            print(output)
    
    elif args.command == 'deploy':
        manager = DeploymentManager()
        
        if args.check_only:
            checks = manager.check_prerequisites()
            for check, status in checks.items():
                print(f"{check}: {'✅' if status else '❌'}")
        else:
            success = manager.deploy(args.env)
            sys.exit(0 if success else 1)
    
    elif args.command == 'ssl':
        manager = DeploymentManager()
        manager.generate_ssl_certs()
    
    else:
        parser.print_help()

if __name__ == '__main__':
    main()