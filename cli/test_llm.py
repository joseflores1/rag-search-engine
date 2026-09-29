from lib.client import get_client


def main() -> None:
    client, model = get_client()
    prompt = "Why is Boot.dev such a great place to learn about RAG? Use one paragraph maximum."

    response = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}]
    )
    if response.usage is None:
        raise RuntimeError("API response has no usage data")

    print(f"Prompt tokens: {response.usage.prompt_tokens}")
    print(f"Response tokens: {response.usage.completion_tokens}")
    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()