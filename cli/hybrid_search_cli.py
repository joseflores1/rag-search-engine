import argparse

from lib.hybrid_search import hybrich_search, normalize_scores, rrf_search_command
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
        "weighted-search", help="Calculate weighted hybrid score given an alpha value"
    )
    weight_search_parser.add_argument("query", type=str, help="Search query")
    weight_search_parser.add_argument(
        "-a", "--alpha", type=float, default=DEFAULT_ALPHA, help="Alpha value"
    )
    weight_search_parser.add_argument(
        "-l",
        "--limit",
        type=int,
        default=DEFAULT_SEARCH_LIMIT,
        help="Maximum number of retrieved results",
    )

    rrf_search__parser = subparsers.add_parser(
        "rrf-search", help="Calculate reciprocal ranked fusion given a search query"
    )
    rrf_search__parser.add_argument("query", type=str, help="Search query")
    rrf_search__parser.add_argument("-k", type=int, default=RRF_K, help="RRF k value")
    rrf_search__parser.add_argument(
        "-l",
        "--limit",
        type=int,
        default=DEFAULT_SEARCH_LIMIT,
        help="Maximum number of retrieved results",
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
            results = hybrich_search(args.query, args.alpha, args.limit)
            for i, res in enumerate(results, 1):
                print(f"{i}. {res['title']}")
                print(f" Hybrid score: {res['hybrid_score']:.3f}")
                print(f" BM25: {res['key_score']:.3f}, Semantic: {res['semantic_score']:.3f}")
                print(f" {res['description'][:DOCUMENT_PREVIEW_LENGTH]}\n")
        case "rrf-search":
            results =rrf_search_command(args.query, args.k, args.limit)            
            for i, res in enumerate(results, 1):
                print(f"{i}. {res['title']}")
                print(f" RRF Score: {res['rrf_score']:.3f}")
                print(
                    f" BM25 Rank: {res['bm25_rank']}, Semantic Rank: {res['chunk_rank']}"  # noqa: E501
                )
                print(f" {res['description'][:DOCUMENT_PREVIEW_LENGTH]}\n")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
