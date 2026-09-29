import argparse

from lib.augmented_generation import (
    citations_command,
    question_command,
    rag_command,
    summarize_command,
)
from lib.search_utils import DEFAULT_SEARCH_LIMIT


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser("rag", help="Perform RAG (search + generate answer)")
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    summarize_parser = subparsers.add_parser(
        "summarize", help="Perform LLM summarization (search + generate summary)"
    )
    summarize_parser.add_argument("query", type=str, help="Search query for summarization")

    summarize_parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_SEARCH_LIMIT,
        help="Maximum number of documents to summarize",
    )

    citations_parser = subparsers.add_parser(
        "citations", help="Generate answer with citations" 
    )
    citations_parser.add_argument("query", type=str, help="Search query for answer generation")
    citations_parser.add_argument(
        "--limit", type=int, default=DEFAULT_SEARCH_LIMIT, help="Maximum number of documents to use"
    )

    question_parser = subparsers.add_parser(
        "question", help="Generate answer to question" 
    )
    question_parser.add_argument("question", type=str, help="Search query for answer generation")
    question_parser.add_argument(
        "--limit", type=int, default=DEFAULT_SEARCH_LIMIT, help="Maximum number of documents to use"
    )

    args = parser.parse_args()

    match args.command:
        case "rag":
            result = rag_command(args.query)
            print("Search Results:")
            for doc in result["search_results"]:
                print(f"- {doc['title']}")
            print()
            print("RAG Response:")
            print(result.get("answer", result.get("error")))

        case "summarize":
            result = summarize_command(args.query, args.limit)
            print("Search Results:")
            for doc in result.get("search_results", []):
                print(f"- {doc['title']}")
            print()
            print("LLM Summary:")
            print(result.get("summary", result.get("error")))

        case "citations":
            result = citations_command(args.query, args.limit)
            print("Search Results:")
            for doc in result.get("search_results", []):
                print(f"- {doc['title']}")
            print()
            print("LLM Answer:")
            print(result.get("answer", result.get("error")))           

        case "question":
            result = question_command(args.question, args.limit)
            print("Search Results:")
            for doc in result.get("search_results", []):
                print(f"- {doc['title']}")
            print()
            print("Answer:")
            print(result.get("answer", result.get("error")))  

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
