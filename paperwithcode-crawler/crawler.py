import requests
from tqdm import tqdm


class PapersWithCodeCrawler:
    """A simple PapersWithCode Crawler."""

    def __init__(self) -> None:
        self.SEARCH_API_URL = f"https://paperswithcode.com/api/v1/search"
        self.PAPER_API_URL = f"https://paperswithcode.com/api/v1/papers/{{paper}}/repositories"

    def get_paper_from_paperswithcode(self, q: str, items_per_page=10, verbose=False):
        """Get list of papers with a keyword q."""
        response = requests.get(self.SEARCH_API_URL, {
            "q": q,
            "items_per_page": items_per_page
        })

        paper_list = []

        response_dict = response.json()

        results_count = response_dict.get("count")

        if results_count > 0:
            result_tqdm = tqdm(response_dict.get("results")
                               ) if verbose else response_dict.get("results")

            for result in result_tqdm:
                paper = result.get("paper")

                paper_list.append({
                    "id": paper.get("id"),
                    "title": paper.get("title")
                })

        if verbose:
            print(f"Fetched {results_count} papers.")

        return paper_list

    def get_repositories_from_paper(self, paper: str, items_per_page=10, verbose=False):
        """Get repositories list from a paper site."""
        response = requests.get(self.PAPER_API_URL.format(paper=paper), {
            "items_per_page": items_per_page
        })

        repositories_list = []

        response_dict = response.json()

        results_count = response_dict.get("count")

        if results_count > 0:
            result_tqdm = tqdm(response_dict.get("results")
                               ) if verbose else response_dict.get("results")

            for result in result_tqdm:
                url = result.get("url")
                is_official = result.get("is_official")

                repositories_list.append(
                    {"url": url, "is_official": is_official})

        if verbose:
            print(f"Fetched {results_count} repositories.")

        return repositories_list

    def get(self, q: str, paper_per_page=10, repo_per_page=10, verbose=False):
        """A magic to get all needed informations."""
        paper_list = self.get_paper_from_paperswithcode(
            q=q, items_per_page=paper_per_page, verbose=verbose)

        paper_and_repos = []

        for paper in paper_list:
            repos = self.get_repositories_from_paper(
                paper.get("id"), items_per_page=repo_per_page, verbose=verbose)
            paper_and_repos.append({
                "paper": paper,
                "repos": repos
            })

        return paper_and_repos
