from pathlib import Path
from urllib.parse import urlparse

from git import Repo


BASE_REPO_DIR = Path("data/repos")


def get_repo_name(github_url: str) -> str:
    """
    Extract repository name from a GitHub URL.

    Example:
    https://github.com/user/my-project
    -> my-project
    """

    path = urlparse(github_url).path.strip("/")

    repo_name = path.split("/")[-1]

    if repo_name.endswith(".git"):
        repo_name = repo_name[:-4]

    return repo_name


def clone_repository(github_url: str) -> Path:
    """
    Clone a GitHub repository into data/repos/.
    """

    repo_name = get_repo_name(github_url)

    repo_path = BASE_REPO_DIR / repo_name

    BASE_REPO_DIR.mkdir(parents=True, exist_ok=True)

    if repo_path.exists():
        print(f"Repository already exists: {repo_path}")
        return repo_path

    print(f"Cloning repository...")
    print(f"URL: {github_url}")
    print(f"Destination: {repo_path}")

    Repo.clone_from(github_url, repo_path)

    print("Repository cloned successfully!")

    return repo_path


if __name__ == "__main__":

    github_url = input("Enter GitHub repository URL: ").strip()

    try:
        repo_path = clone_repository(github_url)

        print(f"\nRepository location:")
        print(repo_path)

    except Exception as e:
        print(f"\nError: {e}")