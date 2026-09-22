import argparse

from lib.hybrid_search import normalize_scores, rrf_search_command, weighted_search_command
from lib.search_utils import DEFAULT_ALPHA, DEFAULT_SEARCH_LIMIT, DOCUMENT_PREVIEW_LENGTH, RRF_K


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser(
        "normalize", help="Normalize a list of scores with min-max"
    )
    normalize_parser.add_argument(
        "scores", nargs="*", type=float, help="List of scores to normalize"
    )

    weight_search_parser = subparsers.add_parser(
        "weighted-search", help="Perform weighted hybrid search"
    )
    weight_search_parser.add_argument("query", type=str, help="Search query")
    weight_search_parser.add_argument(
        "-a",
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        help="Weight for BM25 vs semantic (0-all semantic, 1-all BM25, default=0.5)",
    )
    weight_search_parser.add_argument(
        "-l",
        "--limit",
        type=int,
        default=DEFAULT_SEARCH_LIMIT,
        help="Maximum number of retrieved results (default=5)",
    )

    rrf_search_parser = subparsers.add_parser(
        "rrf-search", help="Perform Reciprocal Rank Fusion search"
    )
    rrf_search_parser.add_argument("query", type=str, help="Search query")
    rrf_search_parser.add_argument(
        "-k",
        type=int,
        default=RRF_K,
        help="RRF k parameter controlling weight distribution (default=60)",
    )
    rrf_search_parser.add_argument(
        "-l",
        "--limit",
        type=int,
        default=DEFAULT_SEARCH_LIMIT,
        help="Maximum number of retrieved results",
    )
    rrf_search_parser.add_argument(
        "--enhance",
        type=str,
        choices=["spell", "rewrite", "expand"],
        help="Query enhancement method",
    )
    rrf_search_parser.add_argument(
        "--rerank-method",
        type=str,
        choices=["individual"],
        help="Method to use for LLM reranking",
    )

    args = parser.parse_args()

    match args.command:
        case "normalize":
            normalized = normalize_scores(args.scores)
            for score in normalized:
                print(f"* {score:.4f}")

        case "weighted-search":
            print(f"Weighted Hybrid Search Results for {args.query}' (alpha={args.alpha}):")
            print(
                f"  Alpha {args.alpha}: {int(args.alpha * 100)}% Keyword, {int((1 - args.alpha) * 100)}% Semantic"  # noqa: E501
            )
            results = weighted_search_command(args.query, args.alpha, args.limit)
            for i, res in enumerate(results, 1):
                print(f"{i}. {res['title']}")
                print(f" Hybrid score: {res['hybrid_score']:.3f}")
                print(f" BM25: {res['key_score']:.3f}, Semantic: {res['semantic_score']:.3f}")
                print(f" {res['description'][:DOCUMENT_PREVIEW_LENGTH]}\n")

        case "rrf-search":
            if args.rerank_method:
                print(f"Re-ranking top {args.limit} results using {args.rerank_method} method...")

            print(f"Reciprocal Rank Fusion Results for '{args.query}' (k={args.k}):")

            enhanced_results = rrf_search_command(
                args.query, args.k, args.limit, args.enhance, args.rerank_method
            )
            enhanced_query = enhanced_results["enhanced_query"]
            results = enhanced_results["results"]
            if args.enhance:
                print(f"Enhanced query ({args.enhance}): '{args.query}' -> '{enhanced_query}'\n")

            for i, res in enumerate(results, 1):
                print(f"{i}. {res['title']}")
                if res["rerank_score"] is not None:
                    print(f"   Re-rank Score: {res['rerank_score']:.3f}/10")
                print(f" RRF Score: {res['rrf_score']:.3f}")
                print(
                    f" BM25 Rank: {res['bm25_rank']}, Semantic Rank: {res['chunk_rank']}"  # noqa: E501
                )
                print(f" {res['description'][:DOCUMENT_PREVIEW_LENGTH]}\n")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
