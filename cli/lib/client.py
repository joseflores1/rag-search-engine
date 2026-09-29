from functools import lru_cache
from typing import TYPE_CHECKING

from dotenv import get_key

from .search_utils import ENV_PATH

if TYPE_CHECKING:
    from openai import OpenAI

_MODEL = "openrouter/free"


@lru_cache
def get_client() -> tuple[OpenAI, str]:
    from openai import OpenAI

    api_key = get_key(ENV_PATH, "OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable not set")

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    return client, _MODEL
