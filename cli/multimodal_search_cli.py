import argparse
import os

from lib.multimodal_search import image_search_command, verify_image_embedding
from lib.search_utils import DOCUMENT_PREVIEW_LENGTH


def main() -> None:
    parser = argparse.ArgumentParser(description="Multimodal Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_parser = subparsers.add_parser("verify_image_embedding", help="Verify image embedding")
    verify_parser.add_argument("image", type=str, help="Image path")

    search_parser = subparsers.add_parser("image_search", help="Search for movies given an image")
    search_parser.add_argument("image", type=str, help="Image path")

    args = parser.parse_args()

    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Image file not found: {args.image}")

    match args.command:
        case "verify_image_embedding":
            verify_image_embedding(args.image) 

        case "image_search":
            results = image_search_command(args.image)
            print(f"Image search results for: {args.image}")
            print("=" * 60)
            for i, res in enumerate(results, 1):
                print(f"{i}. {res["title"]} (similarity: {res['score']})")
                print(f"    {res['document'][:DOCUMENT_PREVIEW_LENGTH]}\n")
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
