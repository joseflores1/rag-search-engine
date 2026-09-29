import argparse
import os

from lib.describe_image import describe_image


def main() -> None:
    parser = argparse.ArgumentParser(description="Multimodal Query Rewriting CLI")
    parser.add_argument("--image", type=str, help="Image file path", required=True)
    parser.add_argument("--query", type=str, help="Query to rewrite", required=True)

    args = parser.parse_args()

    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Image file not found: {args.image}")

    response = describe_image(args.image, args.query)

    rewritten_query = response.choices[0].message.content 
    if rewritten_query is None:
        raise RuntimeError("No text in API response")
    print(f"Rewritten query: {rewritten_query.strip()}")

    if response.usage is not None:
        print(f"Total tokens:   {response.usage.total_tokens}")

if __name__ == "__main__":
    main()
