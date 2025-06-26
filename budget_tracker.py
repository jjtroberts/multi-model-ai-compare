# budget_tracker.py - Complete working implementation
import json
import os
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import logging
from pathlib import Path
import threading
import time

logger = logging.getLogger(__name__)

@dataclass
class UsageRecord:
    """Record of API usage"""
    timestamp: str
    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost: float
    response_time: float
    success: bool
    error: Optional[str] = None

@dataclass
class BudgetLimits:
    """Budget limits configuration"""
    daily_limit: float
    monthly_limit: float
    alert_threshold: float  # Percentage (0.8 = 80%)
    hard_limit: bool = False  # Stop requests when limit reached

class PricingCalculator:
    """Calculate API costs based on current pricing"""
    
    def __init__(self):
        # Pricing per 1K tokens (as of 2024) - Update these regularly
        self.pricing = {
            'claude': {
                'claude-3-5-sonnet-20241022': {'input': 0.003, 'output': 0.015},
                'claude-3-5-haiku-20241022': {'input': 0.00025, 'output': 0.00125},
                'claude-3-opus-20240229': {'input': 0.015, 'output': 0.075},
                # Default fallback for unknown Claude models
                'default': {'input': 0.003, 'output': 0.015}
            },
            'chatgpt': {
                'gpt-4o': {'input': 0.00250, 'output': 0.01000},
                'gpt-4o-mini': {'input': 0.000150, 'output': 0.000600},
                'gpt-4-turbo': {'input': 0.01000, 'output': 0.03000},
                'gpt-4': {'input': 0.03000, 'output': 0.06000},
                # Default fallback
                'default': {'input': 0.00250, 'output': 0.01000}
            },
            'gemini': {
                'gemini-1.5-pro': {'input': 0.000125, 'output': 0.000375},
                'gemini-1.5-flash': {'input': 0.000075, 'output': 0.0003},
                'gemini-pro': {'input': 0.0005, 'output': 0.0015},
                # Default fallback
                'default': {'input': 0.000125, 'output': 0.000375}
            }
        }
    
    def calculate_cost(self, provider: str, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate cost for API usage"""
        try:
            # Get pricing for provider and model
            provider_pricing = self.pricing.get(provider, {})
            model_pricing = provider_pricing.get(model, provider_pricing.get('default', {'input': 0.001, 'output': 0.003}))
            
            # Calculate cost (pricing is per 1K tokens)
            input_cost = (prompt_tokens / 1000) * model_pricing['input']
            output_cost = (completion_tokens / 1000) * model_pricing['output']
            
            return round(input_cost + output_cost, 6)
        except Exception as e:
            logger.error(f"Error calculating cost: {e}")
            # Fallback estimation
            return (prompt_tokens + completion_tokens) / 1000 * 0.002

    def get_model_pricing(self, provider: str, model: str) -> Dict[str, float]:
        """Get pricing information for a model"""
        provider_pricing = self.pricing.get(provider, {})
        return provider_pricing.get(model, provider_pricing.get('default', {'input': 0.001, 'output': 0.003}))

class BudgetTracker:
    """Track API usage and enforce budget limits"""
    
    def __init__(self, db_path: str = "usage_tracking.db"):
        self.db_path = db_path
        self.pricing_calc = PricingCalculator()
        self.lock = threading.Lock()
        self._init_database()
        self.budget_limits = self._load_budget_limits()
    
    def _init_database(self):
        """Initialize SQLite database for usage tracking"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS usage_records (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        provider TEXT NOT NULL,
                        model TEXT NOT NULL,
                        prompt_tokens INTEGER NOT NULL,
                        completion_tokens INTEGER NOT NULL,
                        total_tokens INTEGER NOT NULL,
                        estimated_cost REAL NOT NULL,
                        response_time REAL NOT NULL,
                        success BOOLEAN NOT NULL,
                        error TEXT
                    )
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_timestamp ON usage_records(timestamp)
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_provider ON usage_records(provider)
                """)
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
    
    def _load_budget_limits(self) -> BudgetLimits:
        """Load budget limits from environment variables"""
        return BudgetLimits(
            daily_limit=float(os.getenv('DAILY_BUDGET_USD', '10.00')),
            monthly_limit=float(os.getenv('MONTHLY_BUDGET_USD', '100.00')),
            alert_threshold=float(os.getenv('BUDGET_ALERT_THRESHOLD', '0.8')),
            hard_limit=os.getenv('BUDGET_HARD_LIMIT', 'false').lower() == 'true'
        )
    
    def record_usage(self, provider: str, model: str, prompt_tokens: int, 
                    completion_tokens: int, response_time: float, success: bool,
                    error: Optional[str] = None) -> UsageRecord:
        """Record API usage"""
        
        total_tokens = prompt_tokens + completion_tokens
        estimated_cost = self.pricing_calc.calculate_cost(provider, model, prompt_tokens, completion_tokens)
        
        record = UsageRecord(
            timestamp=datetime.now().isoformat(),
            provider=provider,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            estimated_cost=estimated_cost,
            response_time=response_time,
            success=success,
            error=error
        )
        
        # Save to database
        try:
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("""
                        INSERT INTO usage_records 
                        (timestamp, provider, model, prompt_tokens, completion_tokens, 
                         total_tokens, estimated_cost, response_time, success, error)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        record.timestamp, record.provider, record.model,
                        record.prompt_tokens, record.completion_tokens,
                        record.total_tokens, record.estimated_cost,
                        record.response_time, record.success, record.error
                    ))
        except Exception as e:
            logger.error(f"Failed to record usage: {e}")
        
        logger.info(f"💰 Usage recorded: {provider}/{model} - ${estimated_cost:.4f} ({total_tokens} tokens)")
        return record
    
    def get_daily_usage(self, date: Optional[datetime] = None) -> float:
        """Get total cost for a specific day"""
        if date is None:
            date = datetime.now()
        
        start_date = date.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
        end_date = date.replace(hour=23, minute=59, second=59, microsecond=999999).isoformat()
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                result = conn.execute("""
                    SELECT COALESCE(SUM(estimated_cost), 0) as daily_cost
                    FROM usage_records 
                    WHERE timestamp BETWEEN ? AND ?
                """, (start_date, end_date)).fetchone()
                
                return result[0] if result else 0.0
        except Exception as e:
            logger.error(f"Failed to get daily usage: {e}")
            return 0.0
    
    def get_monthly_usage(self, year: int = None, month: int = None) -> float:
        """Get total cost for a specific month"""
        if year is None or month is None:
            now = datetime.now()
            year = now.year
            month = now.month
        
        start_date = datetime(year, month, 1).isoformat()
        if month == 12:
            end_date = datetime(year + 1, 1, 1).isoformat()
        else:
            end_date = datetime(year, month + 1, 1).isoformat()
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                result = conn.execute("""
                    SELECT COALESCE(SUM(estimated_cost), 0) as monthly_cost
                    FROM usage_records 
                    WHERE timestamp >= ? AND timestamp < ?
                """, (start_date, end_date)).fetchone()
                
                return result[0] if result else 0.0
        except Exception as e:
            logger.error(f"Failed to get monthly usage: {e}")
            return 0.0
    
    def check_budget_status(self) -> Dict:
        """Check current budget status and return warnings if needed"""
        daily_usage = self.get_daily_usage()
        monthly_usage = self.get_monthly_usage()
        
        daily_percentage = (daily_usage / self.budget_limits.daily_limit) if self.budget_limits.daily_limit > 0 else 0
        monthly_percentage = (monthly_usage / self.budget_limits.monthly_limit) if self.budget_limits.monthly_limit > 0 else 0
        
        status = {
            'daily': {
                'used': daily_usage,
                'limit': self.budget_limits.daily_limit,
                'percentage': daily_percentage,
                'remaining': max(0, self.budget_limits.daily_limit - daily_usage),
                'over_limit': daily_usage > self.budget_limits.daily_limit,
                'near_limit': daily_percentage >= self.budget_limits.alert_threshold
            },
            'monthly': {
                'used': monthly_usage,
                'limit': self.budget_limits.monthly_limit,
                'percentage': monthly_percentage,
                'remaining': max(0, self.budget_limits.monthly_limit - monthly_usage),
                'over_limit': monthly_usage > self.budget_limits.monthly_limit,
                'near_limit': monthly_percentage >= self.budget_limits.alert_threshold
            }
        }
        
        return status
    
    def should_block_request(self) -> Tuple[bool, str]:
        """Check if request should be blocked due to budget limits"""
        if not self.budget_limits.hard_limit:
            return False, ""
        
        status = self.check_budget_status()
        
        if status['daily']['over_limit']:
            return True, f"Daily budget exceeded (${status['daily']['used']:.2f} / ${status['daily']['limit']:.2f})"
        
        if status['monthly']['over_limit']:
            return True, f"Monthly budget exceeded (${status['monthly']['used']:.2f} / ${status['monthly']['limit']:.2f})"
        
        return False, ""
    
    def get_usage_summary(self, days: int = 30) -> Dict:
        """Get usage summary for the last N days"""
        start_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                # Total usage
                result = conn.execute("""
                    SELECT 
                        COUNT(*) as total_requests,
                        SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful_requests,
                        SUM(total_tokens) as total_tokens,
                        SUM(estimated_cost) as total_cost,
                        AVG(response_time) as avg_response_time
                    FROM usage_records 
                    WHERE timestamp >= ?
                """, (start_date,)).fetchone()
                
                total_stats = dict(result) if result else {}
                
                # Usage by provider
                provider_stats = {}
                provider_results = conn.execute("""
                    SELECT 
                        provider,
                        COUNT(*) as requests,
                        SUM(total_tokens) as tokens,
                        SUM(estimated_cost) as cost
                    FROM usage_records 
                    WHERE timestamp >= ?
                    GROUP BY provider
                    ORDER BY cost DESC
                """, (start_date,)).fetchall()
                
                for row in provider_results:
                    provider_stats[row['provider']] = dict(row)
                
                return {
                    'period_days': days,
                    'total': total_stats,
                    'by_provider': provider_stats
                }
        except Exception as e:
            logger.error(f"Failed to get usage summary: {e}")
            return {'period_days': days, 'total': {}, 'by_provider': {}}

def render_budget_dashboard(st, tracker: BudgetTracker):
    """Render budget tracking dashboard in Streamlit"""
    
    st.subheader("💰 Budget & Usage Tracking")
    
    # Current budget status
    status = tracker.check_budget_status()
    
    # Daily budget
    col1, col2 = st.columns(2)
    
    with col1:
        daily_color = "red" if status['daily']['over_limit'] else "orange" if status['daily']['near_limit'] else "green"
        st.metric(
            "Daily Usage",
            f"${status['daily']['used']:.2f}",
            f"${status['daily']['remaining']:.2f} remaining"
        )
        
        # Daily progress bar
        daily_progress = min(status['daily']['percentage'], 1.0)
        st.progress(daily_progress)
        st.caption(f"{daily_progress*100:.1f}% of daily budget (${status['daily']['limit']:.2f})")
    
    with col2:
        monthly_color = "red" if status['monthly']['over_limit'] else "orange" if status['monthly']['near_limit'] else "green"
        st.metric(
            "Monthly Usage",
            f"${status['monthly']['used']:.2f}",
            f"${status['monthly']['remaining']:.2f} remaining"
        )
        
        # Monthly progress bar
        monthly_progress = min(status['monthly']['percentage'], 1.0)
        st.progress(monthly_progress)
        st.caption(f"{monthly_progress*100:.1f}% of monthly budget (${status['monthly']['limit']:.2f})")
    
    # Budget alerts
    if status['daily']['over_limit'] or status['monthly']['over_limit']:
        st.error("🚨 Budget limit exceeded! Consider upgrading your plan or reducing usage.")
    elif status['daily']['near_limit'] or status['monthly']['near_limit']:
        st.warning("⚠️ Approaching budget limit. Monitor usage carefully.")
    
    # Usage summary
    with st.expander("📊 Usage Summary", expanded=False):
        summary_period = st.selectbox("Period", [7, 14, 30, 90], index=2, key="budget_period")
        summary = tracker.get_usage_summary(summary_period)
        
        if summary['total'].get('total_requests', 0) > 0:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Requests", summary['total']['total_requests'])
            with col2:
                success_rate = summary['total']['successful_requests'] / summary['total']['total_requests'] * 100
                st.metric("Success Rate", f"{success_rate:.1f}%")
            with col3:
                st.metric("Total Tokens", f"{summary['total']['total_tokens']:,}")
            with col4:
                st.metric("Total Cost", f"${summary['total']['total_cost']:.2f}")
            
            # Usage by provider
            if summary['by_provider']:
                st.write("**Usage by Provider:**")
                for provider, stats in summary['by_provider'].items():
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.write(f"**{provider.title()}**")
                    with col2:
                        st.write(f"{stats['requests']} requests")
                    with col3:
                        st.write(f"${stats['cost']:.2f}")
        else:
            st.info("No usage data available for the selected period.")

def setup_budget_tracking() -> Optional[BudgetTracker]:
    """Setup budget tracking if enabled"""
    if os.getenv('TRACK_USAGE', 'false').lower() == 'true':
        try:
            tracker = BudgetTracker()
            logger.info("💰 Budget tracking enabled")
            return tracker
        except Exception as e:
            logger.error(f"Failed to initialize budget tracking: {e}")
            return None
    return None

def integrate_budget_tracking(original_query_function, tracker: BudgetTracker):
    """Decorator to integrate budget tracking with API queries"""
    
    async def wrapped_query(*args, **kwargs):
        # Check budget before making request
        should_block, block_reason = tracker.should_block_request()
        if should_block:
            # Return a mock response indicating budget limit reached
            class MockResponse:
                def __init__(self):
                    self.model_name = "Budget Limit"
                    self.response = ""
                    self.error = f"Request blocked: {block_reason}"
                    self.response_time = 0.0
                    self.tokens_used = 0
            
            return MockResponse()
        
        # Make the original request
        start_time = time.time()
        response = await original_query_function(*args, **kwargs)
        end_time = time.time()
        
        # Record usage (extract from response or estimate)
        if hasattr(response, 'tokens_used') and response.tokens_used:
            # We have token information
            prompt_tokens = response.tokens_used // 2  # Rough estimate
            completion_tokens = response.tokens_used - prompt_tokens
        else:
            # Estimate tokens from prompt and response length
            prompt = kwargs.get('prompt', args[0] if args else '')
            prompt_tokens = len(prompt.split()) * 1.3  # Rough estimation
            completion_tokens = len(response.response.split()) * 1.3 if response.response else 0
        
        # Extract provider and model from response
        provider = "unknown"
        model = "unknown"
        if hasattr(response, 'model_name'):
            if "Claude" in response.model_name:
                provider = "claude"
            elif "ChatGPT" in response.model_name:
                provider = "chatgpt"
            elif "Gemini" in response.model_name:
                provider = "gemini"
            # Extract model from parentheses if present
            if "(" in response.model_name and ")" in response.model_name:
                model = response.model_name.split("(")[1].split(")")[0]
        
        # Record the usage
        tracker.record_usage(
            provider=provider,
            model=model,
            prompt_tokens=int(prompt_tokens),
            completion_tokens=int(completion_tokens),
            response_time=end_time - start_time,
            success=not bool(response.error),
            error=response.error
        )
        
        return response
    
    return wrapped_query