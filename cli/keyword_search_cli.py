import argparse

from lib.keyword_search import build_command, search_command, tf_command


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    subparsers.add_parser("build", help="Build the inverted index")

    term_parser = subparsers.add_parser("tf", help="Search term frequency given document id and term")
    term_parser.add_argument("id", type=int, help="Document ID")
    term_parser.add_argument("term", type=str, help="Term to search")

    args = parser.parse_args()

    match args.command:
        case "search":
            print("Searching for:", args.query)
            results = search_command(args.query)
            for i, res in enumerate(results, 1):
                print(f"{i}. ({res['id']}) {res['title']}")

        case "build":
            print("Building inverted index...")
            build_command()
            print("Inverted index built successfully.")
        case "tf":
            id, term = args.id, args.term
            print("Retrieving term frequency...")
            tf_command(id, term)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
