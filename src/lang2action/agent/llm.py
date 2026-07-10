"""LLM factory: the model is a config string, the agent is provider-agnostic."""

from langchain.chat_models import init_chat_model

from lang2action.config import Settings


def make_llm(settings: Settings):
    """Build the chat model from an init_chat_model string like
    "openai:gpt-4o-mini" or "anthropic:claude-haiku-4-5" (needs the matching
    provider API key in the environment; tests inject a fake instead)."""
    return init_chat_model(settings.model, temperature=0)
