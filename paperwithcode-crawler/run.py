#!/bin/python3

from argparse import ArgumentParser, BooleanOptionalAction
from crawler import PapersWithCodeCrawler


def main():
    parser = ArgumentParser()
    parser.add_argument("--paper", required=True,
                        help="Paper name (accurately)")
    parser.add_argument("--result", required=True,
                        help="File to store the results")
    parser.add_argument("--max_paper_result", type=int,
                        default=10, help="Max results paper")
    parser.add_argument("--max_repository_result", type=int,
                        default=10, help="Max repositories per paper")
    parser.add_argument("--verbose", default=False,
                        action=BooleanOptionalAction, help="Show progress")

    args = parser.parse_args()

    crawler = PapersWithCodeCrawler()

    result = crawler.get(
        q=args.paper,
        paper_per_page=args.max_paper_result,
        repo_per_page=args.max_repository_result,
        verbose=args.verbose
    )

    with open(args.result, "w+") as f:
        print("="*20, file=f)
        for paper in result:
            title = paper.get("paper").get("title")
            repos = paper.get("repos")

            print(f"{title}", file=f)

            if len(repos) == 0:
                print("This paper doesn't have code implementation.", file=f)
            else:
                for repository in repos:
                    print("URL:", repository.get("url"), file=f)
                    print("Is official?", repository.get("is_official"), file=f)
            print("="*20, file=f)

    print(f"Wrote to {args.result} successfully.")


if __name__ == "__main__":
    main()
