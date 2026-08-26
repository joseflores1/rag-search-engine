import argparse

from lib.search_utils import DEFAULT_CHUNK_SIZE, DEFAULT_SEARCH_LIMIT
from lib.semantic_search import (
    chunk_text,
    embed_query_text,
    embed_text,
    semantic_search,
    verify_embeddings,
    verify_model,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify", help="Verify loading of the embedding model")

    single_embed_parser = subparsers.add_parser(
        "embed_text", help="Generate an embedding for a single text"
    )
    single_embed_parser.add_argument("text", type=str, help="Text to embed")

    subparsers.add_parser(
        "verify_embeddings", help="Verify embeddings for the movei datasetj"
    )

    embed_query_parser = subparsers.add_parser(
        "embed_query", help="Generate an emmbedding for an input query"
    )
    embed_query_parser.add_argument("query", type=str, help="Query to be embed")

    semantic_search_parser = subparsers.add_parser(
        "search", help="Get top [--limit] documents scored by cosine similarity"
    )
    semantic_search_parser.add_argument("query", type=str, help="Search query")
    semantic_search_parser.add_argument("-l", "--limit", type=int, default= DEFAULT_SEARCH_LIMIT, help= "Maximum number of retrieved documents")

    chunk_parser = subparsers.add_parser("chunk", help="Split text into fixed-sized chunks")
    chunk_parser.add_argument("text", type=str, help="Text to chunk")
    chunk_parser.add_argument("-cz", "--chunk-size", type=int, default= DEFAULT_CHUNK_SIZE, help="Size of chunk in words")

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case "verify_embeddings":
            verify_embeddings()
        case "embed_query":
            embed_query_text(args.query)
        case "search":
            semantic_search(args.query, args.limit)
        case "chunk":
            chunk_text(args.text, args.chunk_size)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
