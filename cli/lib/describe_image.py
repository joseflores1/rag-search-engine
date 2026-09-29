import base64
import mimetypes

from lib.client import get_client
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam


def describe_image(image_path: str, query: str) -> ChatCompletion:
    mime, _ = mimetypes.guess_type(image_path)
    mime = mime or "image/jpeg"

    with open(image_path, "rb") as f:
        image_data = f.read()
        
    if len(image_data) == 0:
        raise RuntimeError(f"0 bytes read from {image_path}")

    client, model = get_client()

    prompt = """Given the included image and text query, rewrite the text query to improve search results from a movie database. Make sure to:
    - Synthesize visual and textual information
    - Focus on movie-specific details (actors, scenes, style, etc.)
    - Return only the rewritten query, without any additional commentary
    - Do not return 'User Safety: safe'"""  # noqa: E501



    data_url = f"data:{mime};base64,{base64.b64encode(image_data).decode()}"
    messages: list[ChatCompletionMessageParam] = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt.strip()},
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": query.strip()},
            ],
        }
    ]
    response = client.chat.completions.create(
        model=model, messages=messages
    )
    return response