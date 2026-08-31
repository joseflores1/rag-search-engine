import argparse

from lib.search_utils import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_SEARCH_LIMIT,
    DEFAULT_SEMANTIC_CHUNK_SIZE,
)
from lib.semantic_search import (
    chunk_text,
    embed_chunks,
    embed_query_text,
    embed_text,
    search_chunked,
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
    semantic_search_parser.add_argument(
        "-l",
        "--limit",
        type=int,
        default=DEFAULT_SEARCH_LIMIT,
        help="Maximum number of retrieved documents",
    )

    chunk_parser = subparsers.add_parser(
        "chunk",
        help="Split text into fixed-sized chunks of words with optional overlap",
    )
    chunk_parser.add_argument("text", type=str, help="Text to chunk")
    chunk_parser.add_argument(
        "-cz",
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Maximum size of chunk in words",
    )
    chunk_parser.add_argument(
        "-o",
        "--overlap",
        type=int,
        default=0,
        help="Number of words to overlap between chunks",
    )

    semantic_chunk_parser = subparsers.add_parser(
        "semantic_chunk",
        help="Split text into fixed-sized chunks of sentences with optional overlap",
    )
    semantic_chunk_parser.add_argument("text", type=str, help="Text to chunk")
    semantic_chunk_parser.add_argument(
        "-cz",
        "--max-chunk-size",
        type=int,
        default=DEFAULT_SEMANTIC_CHUNK_SIZE,
        help="Maximum size of chunk in sentences",
    )
    semantic_chunk_parser.add_argument(
        "-o",
        "--overlap",
        type=int,
        default=0,
        help="Number of sentences to overlap between chunks",
    )

    subparsers.add_parser("embed_chunks", help="Generate chunks embeddings")

    search_chunked_parser = subparsers.add_parser(
        "search_chunked", help="Search using chunked embeddings"
    )
    search_chunked_parser.add_argument("query", type=str, help="Search query")
    search_chunked_parser.add_argument(
        "-l",
        "--limit",
        type=int,
        default=DEFAULT_SEARCH_LIMIT,
        help="Maximum number of returned docs",
    )

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
            chunk_text(args.text, False, args.chunk_size, args.overlap)
        case "semantic_chunk":
            chunk_text(args.text, True, args.max_chunk_size, args.overlap)
        case "embed_chunks":
            chunk_embeddings = embed_chunks()
            print(f"Generated {len(chunk_embeddings)} chunked embeddings")
        case "search_chunked":
            result = search_chunked(args.query, args.limit)
            print(f"Query: {result["query"]}")
            print("Results:")
            for i, res in enumerate(result["results"], 1):
                print(f"\n{i}. {res['title']} (score: {res['score']:.4f})")
                print(f"   {res['document']}...")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
