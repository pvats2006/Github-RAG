from pathlib import Path


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
    "artifacts",
}


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
    ".html",
    ".css",
    ".ipynb",
}


def should_ignore(path: Path) -> bool:
    """
    Check whether any part of the path belongs
    to an ignored directory.
    """

    return any(
        part in IGNORED_DIRS
        for part in path.parts
    )


def is_supported_file(path: Path) -> bool:
    """
    Check whether the file extension is supported.
    """

    return path.suffix.lower() in SUPPORTED_EXTENSIONS


def build_repository_structure(repo_path: str | Path) -> dict:
    """
    Build a filtered hierarchical representation
    of the repository.
    """

    repo_dir = Path(repo_path)

    structure = {
        "repository": repo_dir.name,
        "folders": [],
        "files": [],
    }

    for path in sorted(repo_dir.rglob("*")):

        if should_ignore(path):
            continue

        if path.is_dir():
            continue

        if not is_supported_file(path):
            continue

        relative_path = path.relative_to(repo_dir)

        parts = relative_path.parts

        if len(parts) > 1:

            for i in range(1, len(parts)):
                folder = str(
                    Path(*parts[:i])
                )

                if folder not in structure["folders"]:
                    structure["folders"].append(folder)

        structure["files"].append(
            str(relative_path)
        )

    return structure


if __name__ == "__main__":

    repo_path = input(
        "Enter repository path: "
    ).strip()

    structure = build_repository_structure(repo_path)

    print("\nRepository:")
    print(structure["repository"])

    print("\nFolders:")

    for folder in structure["folders"]:
        print(f"  📁 {folder}")

    print("\nFiles:")

    for file in structure["files"]:
        print(f"  📄 {file}")

    print(
        f"\nTotal folders: "
        f"{len(structure['folders'])}"
    )

    print(
        f"Total files: "
        f"{len(structure['files'])}"
    )