# default_models.py - Complete working implementation
import os
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class DefaultModelConfig:
    """Manage default model configurations from environment variables"""
    
    def __init__(self):
        self.default_models = self._load_default_models()
        self.model_preferences = self._load_model_preferences()
    
    def _load_default_models(self) -> Dict[str, str]:
        """Load default models from environment variables"""
        return {
            'claude': os.getenv('DEFAULT_CLAUDE_MODEL', 'claude-3-5-sonnet-20241022'),
            'chatgpt': os.getenv('DEFAULT_OPENAI_MODEL', 'gpt-4o'),
            'gemini': os.getenv('DEFAULT_GEMINI_MODEL', 'gemini-1.5-pro-latest')
        }
    
    def _load_model_preferences(self) -> Dict[str, Dict[str, str]]:
        """Load model preferences for different use cases"""
        return {
            'fast': {
                'claude': os.getenv('FAST_CLAUDE_MODEL', 'claude-3-5-haiku-20241022'),
                'chatgpt': os.getenv('FAST_OPENAI_MODEL', 'gpt-4o-mini'),
                'gemini': os.getenv('FAST_GEMINI_MODEL', 'gemini-1.5-flash-latest')
            },
            'quality': {
                'claude': os.getenv('QUALITY_CLAUDE_MODEL', 'claude-3-5-sonnet-20241022'),
                'chatgpt': os.getenv('QUALITY_OPENAI_MODEL', 'gpt-4o'),
                'gemini': os.getenv('QUALITY_GEMINI_MODEL', 'gemini-1.5-pro-latest')
            },
            'cost_effective': {
                'claude': os.getenv('COST_CLAUDE_MODEL', 'claude-3-5-haiku-20241022'),
                'chatgpt': os.getenv('COST_OPENAI_MODEL', 'gpt-4o-mini'),
                'gemini': os.getenv('COST_GEMINI_MODEL', 'gemini-1.5-flash-latest')
            }
        }
    
    def get_default_model(self, provider: str, use_case: str = 'default') -> Optional[str]:
        """Get default model for a provider and use case"""
        if use_case == 'default':
            return self.default_models.get(provider)
        else:
            return self.model_preferences.get(use_case, {}).get(provider)
    
    def get_all_defaults(self) -> Dict[str, str]:
        """Get all default models"""
        return self.default_models.copy()
    
    def get_preferences_for_use_case(self, use_case: str) -> Dict[str, str]:
        """Get model preferences for a specific use case"""
        return self.model_preferences.get(use_case, {})

def select_models_with_defaults(available_models: Dict[str, Dict[str, str]], 
                               api_keys: Dict[str, str],
                               config: DefaultModelConfig,
                               use_case: str = 'default') -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Select models with smart defaults based on configuration
    
    Args:
        available_models: Available models from API discovery
        api_keys: Available API keys
        config: Default model configuration
        use_case: Use case for model selection (default, fast, quality, cost_effective)
    
    Returns:
        Tuple of (model_selections, model_names) for display
    """
    model_selections = {}
    model_names = {}
    
    for provider in ['claude', 'chatgpt', 'gemini']:
        if not api_keys.get(provider):
            continue
        
        provider_models = available_models.get(provider, {})
        if not provider_models:
            continue
        
        # Get default model for this provider and use case
        default_model_id = config.get_default_model(provider, use_case)
        
        # Find matching model in available models
        selected_model_id = None
        selected_model_name = None
        
        # First, try exact match with default
        if default_model_id:
            for name, model_id in provider_models.items():
                if model_id == default_model_id:
                    selected_model_id = model_id
                    selected_model_name = name
                    logger.info(f"✅ Using configured default for {provider}: {selected_model_name}")
                    break
        
        # If no exact match, try partial match (useful for versioned models)
        if not selected_model_id and default_model_id:
            # Extract base model name (e.g., "claude-3-5-sonnet" from "claude-3-5-sonnet-20241022")
            base_default = default_model_id.split('-')
            if len(base_default) >= 3:
                base_pattern = '-'.join(base_default[:3])  # e.g., "claude-3-5"
                
                for name, model_id in provider_models.items():
                    if base_pattern in model_id:
                        selected_model_id = model_id
                        selected_model_name = name
                        logger.info(f"🔄 Using similar model for {provider}: {selected_model_name} (wanted {default_model_id})")
                        break
        
        # Fallback to first available model
        if not selected_model_id:
            selected_model_name = list(provider_models.keys())[0]
            selected_model_id = provider_models[selected_model_name]
            logger.info(f"⚠️ Using fallback model for {provider}: {selected_model_name}")
        
        model_selections[provider] = selected_model_id
        model_names[provider] = selected_model_name
    
    return model_selections, model_names

def initialize_default_models(api_keys: Dict[str, str]) -> Tuple[Dict[str, str], str]:
    """
    Initialize model selections with defaults
    
    Returns:
        Tuple of (model_selections, status_message)
    """
    config = DefaultModelConfig()
    auto_select = os.getenv('AUTO_SELECT_MODELS', 'true').lower() == 'true'
    default_use_case = os.getenv('DEFAULT_USE_CASE', 'default')
    
    if not auto_select:
        return {}, "Auto-selection disabled"
    
    # This would be called after model discovery
    status_messages = []
    model_selections = {}
    
    for provider in ['claude', 'chatgpt', 'gemini']:
        if api_keys.get(provider):
            default_model = config.get_default_model(provider, default_use_case)
            if default_model:
                model_selections[provider] = default_model
                status_messages.append(f"✅ {provider}: {default_model}")
            else:
                status_messages.append(f"⚠️ {provider}: no default configured")
    
    status = f"Auto-selected models: {', '.join(status_messages)}"
    return model_selections, status

def render_use_case_selector(st, config: DefaultModelConfig) -> str:
    """Render use case selector for model preferences"""
    
    use_cases = {
        'default': '🎯 Default (Balanced)',
        'fast': '⚡ Fast (Quick responses)',
        'quality': '💎 Quality (Best results)',
        'cost_effective': '💰 Cost-effective (Cheapest)'
    }
    
    default_use_case = os.getenv('DEFAULT_USE_CASE', 'default')
    default_index = list(use_cases.keys()).index(default_use_case) if default_use_case in use_cases else 0
    
    selected_use_case = st.selectbox(
        "Model Selection Strategy",
        options=list(use_cases.keys()),
        format_func=lambda x: use_cases[x],
        index=default_index,
        help="Choose optimization strategy for model selection"
    )
    
    # Show what models would be selected
    if selected_use_case != 'default':
        preferences = config.get_preferences_for_use_case(selected_use_case)
        with st.expander(f"Models for {use_cases[selected_use_case]}", expanded=False):
            for provider, model in preferences.items():
                if model:
                    st.write(f"• **{provider.title()}**: {model}")
    
    return selected_use_case

def render_model_selection_with_defaults(st, provider: str, available_models: Dict[str, str], 
                                       config: DefaultModelConfig, use_case: str = 'default') -> str:
    """Render model selection with smart defaults"""
    
    if not available_models:
        st.error(f"❌ No {provider} models available")
        return None
    
    # Get default model
    default_model_id = config.get_default_model(provider, use_case)
    default_index = 0
    
    # Find index of default model
    if default_model_id:
        model_ids = list(available_models.values())
        try:
            default_index = model_ids.index(default_model_id)
        except ValueError:
            # Try partial match
            for i, model_id in enumerate(model_ids):
                if default_model_id in model_id or model_id in default_model_id:
                    default_index = i
                    break
    
    # Model selectbox with default pre-selected
    model_names = list(available_models.keys())
    selected_name = st.selectbox(
        f"{provider.title()} Model",
        options=model_names,
        index=default_index,
        help=f"Default: {default_model_id}" if default_model_id else "No default configured"
    )
    
    return available_models[selected_name]

# Utility functions for checking and validating defaults
def validate_default_models() -> Dict[str, bool]:
    """Validate that default model environment variables are set"""
    config = DefaultModelConfig()
    defaults = config.get_all_defaults()
    
    validation = {}
    for provider, model in defaults.items():
        # Check if it's using the hardcoded default or a custom env var
        env_var = f"DEFAULT_{provider.upper()}_MODEL"
        is_custom = os.getenv(env_var) is not None
        validation[provider] = {
            'model': model,
            'env_var': env_var,
            'is_custom': is_custom,
            'is_set': bool(model)
        }
    
    return validation

def get_model_info_for_provider(provider: str, use_case: str = 'default') -> Dict[str, str]:
    """Get model information for a specific provider and use case"""
    config = DefaultModelConfig()
    
    if use_case == 'default':
        return {
            'model': config.get_default_model(provider, use_case),
            'env_var': f"DEFAULT_{provider.upper()}_MODEL",
            'use_case': use_case
        }
    else:
        return {
            'model': config.get_default_model(provider, use_case),
            'env_var': f"{use_case.upper()}_{provider.upper()}_MODEL",
            'use_case': use_case
        }

ENV_TEMPLATE_WITH_DEFAULTS = """
# Multi-Model AI Comparison Tool Configuration
ANTHROPIC_API_KEY=your_claude_api_key_here
OPENAI_API_KEY=your_openai_api_key_here  
GOOGLE_API_KEY=your_google_api_key_here

DEFAULT_CLAUDE_MODEL=claude-3-5-sonnet-20241022
DEFAULT_OPENAI_MODEL=gpt-4o
DEFAULT_GEMINI_MODEL=gemini-1.5-pro-latest

TRACK_USAGE=true
DAILY_BUDGET_USD=10.00
MONTHLY_BUDGET_USD=100.00
AUTO_SELECT_MODELS=true
"""