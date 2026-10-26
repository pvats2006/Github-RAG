from pathlib import Path


# Files/directories we never want to index
IGNORED_DIRS = {
    ".git",
    ".github",
    "__pycache__",
    "node_modules",
    ".venv",
    "venv",
    "dist",
    "build",
    "coverage",
    "catboost_info",
    ".ipynb_checkpoints",
}

# File extensions that are useful for our RAG system
SUPPORTED_EXTENSIONS = {
    ".md",
    ".txt",
    ".py",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".java",
    ".cpp",
    ".cc",
    ".c",
    ".h",
    ".hpp",
    ".go",
    ".rs",
    ".php",
    ".rb",
    ".swift",
    ".kt",
    ".kts",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
}


def should_ignore(path: Path) -> bool:
    """
    Check whether a file belongs to an ignored directory.
    """

    return any(part in IGNORED_DIRS for part in path.parts)


def scan_repository(repo_path: str | Path):
    """
    Scan repository and return supported files.
    """

    repo_dir = Path(repo_path)

    files = []

    for path in repo_dir.rglob("*"):

        if not path.is_file():
            continue

        if should_ignore(path):
            continue

        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        files.append(path)

    return files


if __name__ == "__main__":

    repo_path = input("Enter repository path: ").strip()

    files = scan_repository(repo_path)

    print(f"\nFound {len(files)} supported files:\n")

    for file in files:
        print(file)