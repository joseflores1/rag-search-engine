import argparse

from lib.keyword_search import (
    build_command,
    idf_command,
    search_command,
    tf_command,
    tf_idf_command,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    subparsers.add_parser("build", help="Build the inverted index")

    term_parser = subparsers.add_parser(
        "tf", help="Get term frequency given document id and term"
    )
    term_parser.add_argument("doc_id", type=int, help="Document ID")
    term_parser.add_argument("term", type=str, help="Term to get the frequency for")

    idf_parser = subparsers.add_parser(
        "idf", help="Calculate Inverse Document Frequency value given a term"
    )
    idf_parser.add_argument("term", type=str, help="Term to calculate the IDF for")

    tf_idf_parser = subparsers.add_parser("tfidf", help="Calculate the TF-IDF value given a document ID and term")
    tf_idf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_idf_parser.add_argument("term", type=str, help="Term to calculate the TF-IDF for")

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
            print("Retrieving term frequency...")
            tf = tf_command(args.doc_id, args.term)
            print(f"'{args.term}' appears {tf} times in document {args.doc_id}")
        case "idf":
            print("Calculating IDF...")
            idf = idf_command(args.term)
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
        case "tfidf":
            print("Calculating TF-IDF...")
            tf_idf = tf_idf_command(args.doc_id, args.term)
            print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
