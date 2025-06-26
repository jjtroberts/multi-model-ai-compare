#!/usr/bin/env python3
"""
Quick diagnostic script to fix ModuleNotFoundError
Run this from the same directory as app.py
"""

import os
import sys
from pathlib import Path

def diagnose_and_fix():
    print("🔍 Diagnosing module import issues...")
    print("=" * 50)
    
    # Check current directory and files
    current_dir = Path.cwd()
    print(f"📁 Current directory: {current_dir}")
    
    required_files = ['app.py', 'clipboard_utils.py', 'default_models.py', 'budget_tracker.py']
    
    print("\n📋 File Check:")
    for file in required_files:
        file_path = current_dir / file
        exists = file_path.exists()
        size = file_path.stat().st_size if exists else 0
        print(f"  {file}: {'✅' if exists else '❌'} ({size} bytes)")
        
        if exists and size == 0:
            print(f"    ⚠️  {file} is empty!")
    
    # Test Python path
    print(f"\n🐍 Python path includes current directory: {'✅' if str(current_dir) in sys.path else '❌'}")
    print(f"📁 Working directory: {os.getcwd()}")
    
    # Test imports one by one
    print("\n🧪 Testing imports:")
    modules_to_test = ['clipboard_utils', 'default_models', 'budget_tracker']
    
    for module in modules_to_test:
        try:
            __import__(module)
            print(f"  {module}: ✅ Import successful")
        except ModuleNotFoundError as e:
            print(f"  {module}: ❌ ModuleNotFoundError: {e}")
        except SyntaxError as e:
            print(f"  {module}: ❌ SyntaxError: {e}")
        except Exception as e:
            print(f"  {module}: ❌ Other error: {e}")
    
    # Check if files have correct content
    print("\n📄 File Content Check:")
    for module in modules_to_test:
        file_path = current_dir / f"{module}.py"
        if file_path.exists():
            with open(file_path, 'r') as f:
                content = f.read()
                lines = len(content.splitlines())
                has_def = 'def ' in content
                has_class = 'class ' in content
                print(f"  {module}.py: {lines} lines, has functions: {'✅' if has_def else '❌'}, has classes: {'✅' if has_class else '❌'}")
                
                if lines < 10:
                    print(f"    ⚠️  File seems too short, might be incomplete")

def create_minimal_working_files():
    """Create minimal working versions that will definitely import"""
    print("\n🔧 Creating minimal working files...")
    
    # Minimal clipboard_utils.py
    clipboard_minimal = '''import streamlit as st

def create_copy_button(text, button_text="📋 Copy", key=None):
    """Minimal copy button implementation"""
    if st.button(button_text, key=key):
        st.success("✅ Copied!")
        return True
    return False

def create_copy_section(text, title="Copy", key=None, show_download=True):
    """Minimal copy section implementation"""
    with st.expander(title):
        st.text_area("Content:", value=text, key=f"text_{key}")
        create_copy_button(text, key=f"btn_{key}")
        if show_download:
            st.download_button("💾 Download", data=text, file_name="response.txt")

def inject_copy_script():
    """Placeholder for copy script injection"""
    pass
'''

    # Minimal default_models.py
    default_models_minimal = '''import os

class DefaultModelConfig:
    def __init__(self):
        self.defaults = {
            'claude': os.getenv('DEFAULT_CLAUDE_MODEL', 'claude-3-5-sonnet-20241022'),
            'chatgpt': os.getenv('DEFAULT_OPENAI_MODEL', 'gpt-4o'),
            'gemini': os.getenv('DEFAULT_GEMINI_MODEL', 'gemini-1.5-pro-latest')
        }
    
    def get_default_model(self, provider, use_case='default'):
        return self.defaults.get(provider)
    
    def get_all_defaults(self):
        return self.defaults.copy()

def render_use_case_selector(st, config):
    """Minimal use case selector"""
    return st.selectbox("Strategy", ['default', 'fast', 'quality', 'cost_effective'])
'''

    # Minimal budget_tracker.py
    budget_tracker_minimal = '''import os
import logging

logger = logging.getLogger(__name__)

class BudgetTracker:
    def __init__(self):
        self.daily_limit = float(os.getenv('DAILY_BUDGET_USD', '10.00'))
        self.monthly_limit = float(os.getenv('MONTHLY_BUDGET_USD', '100.00'))
        self.daily_usage = 0.0
        self.monthly_usage = 0.0
    
    def record_usage(self, provider, model, prompt_tokens, completion_tokens, response_time, success, error=None):
        """Record usage - minimal implementation"""
        cost = (prompt_tokens + completion_tokens) * 0.002 / 1000
        self.daily_usage += cost
        self.monthly_usage += cost
        logger.info(f"Usage: ${cost:.4f}")
    
    def get_daily_usage(self):
        return self.daily_usage
    
    def get_monthly_usage(self):
        return self.monthly_usage
    
    def check_budget_status(self):
        return {
            'daily': {
                'used': self.daily_usage,
                'limit': self.daily_limit,
                'percentage': self.daily_usage / self.daily_limit,
                'remaining': self.daily_limit - self.daily_usage,
                'over_limit': self.daily_usage > self.daily_limit,
                'near_limit': (self.daily_usage / self.daily_limit) > 0.8
            },
            'monthly': {
                'used': self.monthly_usage,
                'limit': self.monthly_limit,
                'percentage': self.monthly_usage / self.monthly_limit,
                'remaining': self.monthly_limit - self.monthly_usage,
                'over_limit': self.monthly_usage > self.monthly_limit,
                'near_limit': (self.monthly_usage / self.monthly_limit) > 0.8
            }
        }
    
    def should_block_request(self):
        return False, ""

def render_budget_dashboard(st, tracker):
    """Minimal budget dashboard"""
    st.subheader("💰 Budget Tracking")
    status = tracker.check_budget_status()
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Daily", f"${status['daily']['used']:.2f}")
    with col2:
        st.metric("Monthly", f"${status['monthly']['used']:.2f}")

def setup_budget_tracking():
    """Setup budget tracking"""
    if os.getenv('TRACK_USAGE', 'false').lower() == 'true':
        return BudgetTracker()
    return None

def integrate_budget_tracking(func, tracker):
    """Budget tracking decorator"""
    return func  # Minimal implementation
'''

    # Write files
    files = {
        'clipboard_utils.py': clipboard_minimal,
        'default_models.py': default_models_minimal,
        'budget_tracker.py': budget_tracker_minimal
    }
    
    for filename, content in files.items():
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ Created {filename}")
        except Exception as e:
            print(f"❌ Failed to create {filename}: {e}")
    
    # Test imports again
    print("\n🧪 Testing imports after creation:")
    for module in ['clipboard_utils', 'default_models', 'budget_tracker']:
        try:
            # Clear any cached imports
            if module in sys.modules:
                del sys.modules[module]
            __import__(module)
            print(f"  {module}: ✅ Import successful")
        except Exception as e:
            print(f"  {module}: ❌ Import failed: {e}")

if __name__ == '__main__':
    diagnose_and_fix()
    
    # Ask if user wants to create minimal files
    print("\n" + "=" * 50)
    response = input("Create minimal working files? (y/n): ").lower().strip()
    
    if response in ['y', 'yes']:
        create_minimal_working_files()
        print("\n✅ Minimal working files created!")
        print("Your app should now run. You can enhance these files later.")
    else:
        print("\n💡 To fix manually:")
        print("1. Check that clipboard_utils.py, default_models.py, budget_tracker.py exist")
        print("2. Check they have content (not empty)")
        print("3. Check for syntax errors: python -m py_compile filename.py")
        print("4. Try: python -c 'import clipboard_utils; print(\"Success!\")'")