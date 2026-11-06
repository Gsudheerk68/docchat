"""LLM factory: one place to switch between providers via .env."""
from functools import lru_cache

from app import config


@lru_cache
def get_llm():
    provider = config.LLM_PROVIDER
    model = config.LLM_MODEL or config.DEFAULT_MODELS.get(provider)

    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model, temperature=config.TEMPERATURE)
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model, temperature=config.TEMPERATURE)
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, temperature=config.TEMPERATURE)
    if provider == "deepseek":  # OpenAI-compatible API
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, temperature=config.TEMPERATURE,
                          api_key=config.DEEPSEEK_API_KEY, base_url=config.DEEPSEEK_BASE_URL)
    if provider == "ollama":  # free, runs on your own machine
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, temperature=config.TEMPERATURE,
                          api_key="ollama", base_url=config.OLLAMA_BASE_URL)
    raise ValueError(f"Unknown LLM_PROVIDER '{provider}'. Use google, anthropic, openai, deepseek or ollama.")
