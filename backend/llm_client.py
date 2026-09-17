"""
Universal LLM Client
────────────────────
Supports multiple LLM providers through a unified interface.
Configure via .env: LLM_PROVIDER=nemotron|openai|anthropic|gemini|ollama|openrouter
"""

import os
import json
import requests
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from utils.logger import get_logger

logger = get_logger("llm_client")


@dataclass
class LLMResponse:
    content: str
    provider: str
    model: str
    usage: Optional[Dict[str, int]] = None
    error: Optional[str] = None


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        pass


class NemotronProvider(LLMProvider):
    """NVIDIA Nemotron via NVIDIA API."""
    
    def __init__(self):
        self.api_key = os.getenv("NEMOTRON_API_KEY") or os.getenv("NVIDIA_API_KEY")
        self.endpoint = "https://integrate.api.nvidia.com/v1/chat/completions"
        self.default_model = "nvidia/nemotron-3-ultra"
    
    def get_name(self) -> str:
        return "nemotron"
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        if not self.api_key:
            return LLMResponse("", "nemotron", self.default_model, error="NEMOTRON_API_KEY not set")
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": kwargs.get("model", self.default_model),
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.3),
            "max_tokens": kwargs.get("max_tokens", 1024),
            "stream": False
        }
        
        try:
            resp = requests.post(self.endpoint, headers=headers, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            return LLMResponse(
                content=data["choices"][0]["message"]["content"],
                provider="nemotron",
                model=data.get("model", self.default_model),
                usage=data.get("usage")
            )
        except Exception as e:
            logger.error(f"Nemotron error: {e}")
            return LLMResponse("", "nemotron", self.default_model, error=str(e))


class OpenAIProvider(LLMProvider):
    """OpenAI GPT models."""
    
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.endpoint = "https://api.openai.com/v1/chat/completions"
        self.default_model = "gpt-4o-mini"
    
    def get_name(self) -> str:
        return "openai"
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        if not self.api_key:
            return LLMResponse("", "openai", self.default_model, error="OPENAI_API_KEY not set")
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": kwargs.get("model", self.default_model),
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.3),
            "max_tokens": kwargs.get("max_tokens", 1024)
        }
        
        try:
            resp = requests.post(self.endpoint, headers=headers, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            return LLMResponse(
                content=data["choices"][0]["message"]["content"],
                provider="openai",
                model=data.get("model", self.default_model),
                usage=data.get("usage")
            )
        except Exception as e:
            logger.error(f"OpenAI error: {e}")
            return LLMResponse("", "openai", self.default_model, error=str(e))


class AnthropicProvider(LLMProvider):
    """Anthropic Claude models."""
    
    def __init__(self):
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        self.endpoint = "https://api.anthropic.com/v1/messages"
        self.default_model = "claude-3-5-sonnet-20241022"
    
    def get_name(self) -> str:
        return "anthropic"
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        if not self.api_key:
            return LLMResponse("", "anthropic", self.default_model, error="ANTHROPIC_API_KEY not set")
        
        # Convert messages format for Anthropic
        system_msg = ""
        user_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_msg = msg["content"]
            else:
                user_messages.append(msg)
        
        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        payload = {
            "model": kwargs.get("model", self.default_model),
            "messages": user_messages,
            "temperature": kwargs.get("temperature", 0.3),
            "max_tokens": kwargs.get("max_tokens", 1024)
        }
        if system_msg:
            payload["system"] = system_msg
        
        try:
            resp = requests.post(self.endpoint, headers=headers, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            return LLMResponse(
                content=data["content"][0]["text"],
                provider="anthropic",
                model=data.get("model", self.default_model),
                usage=data.get("usage")
            )
        except Exception as e:
            logger.error(f"Anthropic error: {e}")
            return LLMResponse("", "anthropic", self.default_model, error=str(e))


class GeminiProvider(LLMProvider):
    """Google Gemini models."""
    
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.endpoint = "https://generativelanguage.googleapis.com/v1beta/models"
        self.default_model = "gemini-1.5-flash"
    
    def get_name(self) -> str:
        return "gemini"
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        if not self.api_key:
            return LLMResponse("", "gemini", self.default_model, error="GEMINI_API_KEY not set")
        
        model = kwargs.get("model", self.default_model)
        url = f"{self.endpoint}/{model}:generateContent?key={self.api_key}"
        
        # Convert messages format for Gemini
        contents = []
        for msg in messages:
            role = "user" if msg["role"] != "assistant" else "model"
            contents.append({"role": role, "parts": [{"text": msg["content"]}]})
        
        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": kwargs.get("temperature", 0.3),
                "maxOutputTokens": kwargs.get("max_tokens", 1024)
            }
        }
        
        try:
            resp = requests.post(url, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            content = data["candidates"][0]["content"]["parts"][0]["text"]
            return LLMResponse(
                content=content,
                provider="gemini",
                model=model,
                usage=data.get("usageMetadata")
            )
        except Exception as e:
            logger.error(f"Gemini error: {e}")
            return LLMResponse("", "gemini", self.default_model, error=str(e))


class OllamaProvider(LLMProvider):
    """Local Ollama models (free, self-hosted)."""
    
    def __init__(self):
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.default_model = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    
    def get_name(self) -> str:
        return "ollama"
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        model = kwargs.get("model", self.default_model)
        url = f"{self.base_url}/api/chat"
        
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", 0.3),
                "num_predict": kwargs.get("max_tokens", 1024)
            }
        }
        
        try:
            resp = requests.post(url, json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            return LLMResponse(
                content=data["message"]["content"],
                provider="ollama",
                model=model,
                usage={"prompt_tokens": data.get("prompt_eval_count", 0), 
                       "completion_tokens": data.get("eval_count", 0)}
            )
        except Exception as e:
            logger.error(f"Ollama error: {e}")
            return LLMResponse("", "ollama", self.default_model, error=str(e))


class OpenRouterProvider(LLMProvider):
    """OpenRouter - access to 100+ models (free tier available)."""
    
    def __init__(self):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.endpoint = "https://openrouter.ai/api/v1/chat/completions"
        self.default_model = "meta-llama/llama-3.1-8b-instruct:free"
    
    def get_name(self) -> str:
        return "openrouter"
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        if not self.api_key:
            return LLMResponse("", "openrouter", self.default_model, error="OPENROUTER_API_KEY not set")
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/ai-trader",
            "X-Title": "AI Trader"
        }
        payload = {
            "model": kwargs.get("model", self.default_model),
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.3),
            "max_tokens": kwargs.get("max_tokens", 1024)
        }
        
        try:
            resp = requests.post(self.endpoint, headers=headers, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            return LLMResponse(
                content=data["choices"][0]["message"]["content"],
                provider="openrouter",
                model=data.get("model", self.default_model),
                usage=data.get("usage")
            )
        except Exception as e:
            logger.error(f"OpenRouter error: {e}")
            return LLMResponse("", "openrouter", self.default_model, error=str(e))


# Provider registry
PROVIDERS = {
    "nemotron": NemotronProvider,
    "nvidia": NemotronProvider,
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "claude": AnthropicProvider,
    "gemini": GeminiProvider,
    "google": GeminiProvider,
    "ollama": OllamaProvider,
    "openrouter": OpenRouterProvider,
}


class LLMClient:
    """Universal LLM client with automatic provider selection."""
    
    def __init__(self, provider: Optional[str] = None, model: Optional[str] = None):
        self.provider_name = provider or os.getenv("LLM_PROVIDER", "openrouter").lower()
        self.model = model
        self._provider: Optional[LLMProvider] = None
        self._init_provider()
    
    def _init_provider(self):
        provider_class = PROVIDERS.get(self.provider_name)
        if not provider_class:
            available = ", ".join(PROVIDERS.keys())
            raise ValueError(f"Unknown provider: {self.provider_name}. Available: {available}")
        self._provider = provider_class()
        logger.info(f"LLM Client initialized: {self._provider.get_name()}")
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """Send chat messages to the configured LLM provider."""
        if self.model:
            kwargs["model"] = self.model
        return self._provider.chat(messages, **kwargs)
    
    def ask(self, prompt: str, system: Optional[str] = None, **kwargs) -> LLMResponse:
        """Simple single-prompt interface."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.chat(messages, **kwargs)
    
    def get_provider_name(self) -> str:
        return self._provider.get_name() if self._provider else "unknown"


# Convenience function
def get_llm_client(provider: Optional[str] = None, model: Optional[str] = None) -> LLMClient:
    """Get a configured LLM client instance."""
    return LLMClient(provider, model)


# Trading-specific prompts
TRADING_SYSTEM_PROMPT = """You are an expert quantitative trading analyst for Indian NIFTY options.
Provide concise, actionable insights. Focus on risk management and probability.
Never give financial advice - only analysis based on provided data.
Format: 3 bullet points max, each under 160 chars."""


def analyze_market_state(
    nifty_price: float,
    regime: str,
    positions: List[Dict],
    recent_trades: List[Dict],
    provider: Optional[str] = None
) -> LLMResponse:
    """Analyze current market state using LLM."""
    client = get_llm_client(provider)
    
    prompt = f"""
Market Snapshot:
- NIFTY: {nifty_price:.1f}
- Regime: {regime}
- Open Positions: {len(positions)}
- Recent Trades (last 5): {len(recent_trades)}

Positions:
{json.dumps(positions[:3], default=str) if positions else "None"}

Recent Trades:
{json.dumps(recent_trades[:5], default=str) if recent_trades else "None"}

Provide:
1. Risk assessment (1 line)
2. Key opportunity/concern (1 line)  
3. Suggested action (1 line)
"""
    
    return client.ask(prompt, system=TRADING_SYSTEM_PROMPT, temperature=0.2, max_tokens=300)


def analyze_trade_journey(journey: List[Dict], entry_data: Dict) -> LLMResponse:
    """Analyze a completed trade's journey for learning."""
    client = get_llm_client(provider)
    
    prompt = f"""
Trade Analysis:
Entry: {entry_data.get('symbol')} {entry_data.get('direction')} @ ₹{entry_data.get('entry_premium')}
Strategy: {entry_data.get('strategy')} | Regime: {entry_data.get('regime')}
Exit: {entry_data.get('exit_reason')} @ ₹{entry_data.get('exit_premium')} | P&L: ₹{entry_data.get('realised_pnl')}

Journey ({len(journey)} points):
{json.dumps(journey[-10:], default=str) if journey else "No journey data"}

What went well? What to improve? One key lesson.
"""
    
    return client.ask(prompt, system=TRADING_SYSTEM_PROMPT, temperature=0.2, max_tokens=300)


if __name__ == "__main__":
    # Test the client
    import sys
    provider = sys.argv[1] if len(sys.argv) > 1 else None
    client = get_llm_client(provider)
    
    print(f"Testing {client.get_provider_name()}...")
    resp = client.ask("Say 'OK' if you're working", temperature=0)
    print(f"Response: {resp.content}")
    if resp.error:
        print(f"Error: {resp.error}")