from pathlib import Path


LANGUAGE_MAP = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".java": "java",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".c": "c",
    ".h": "c",
    ".hpp": "cpp",
    ".go": "go",
    ".rs": "rust",
    ".php": "php",
    ".rb": "ruby",
    ".swift": "swift",
    ".kt": "kotlin",
    ".md": "markdown",
    ".txt": "text",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
}


def get_language(path: Path) -> str:
    """
    Detect programming/documentation language
    from the file extension.
    """

    return LANGUAGE_MAP.get(
        path.suffix.lower(),
        "unknown"
    )


from typing import Optional


def extract_metadata(file_path: Path, repo_path: Path) -> Optional[dict]:
    """
    Read a file and create a metadata object.
    """

    try:
        content = file_path.read_text(
            encoding="utf-8",
            errors="ignore"
        )
    except Exception as e:
        print(f"Could not read {file_path}: {e}")
        return None

    relative_path = file_path.relative_to(repo_path)

    lines = content.splitlines()

    metadata = {
        "file_path": str(relative_path),
        "file_name": file_path.name,
        "extension": file_path.suffix.lower(),
        "language": get_language(file_path),
        "content": content,
        "line_start": 1,
        "line_end": len(lines),
    }

    return metadata


if __name__ == "__main__":

    repo_path = Path(
        input("Enter repository path: ").strip()
    )

    test_file = repo_path / "README.md"

    metadata = extract_metadata(
        test_file,
        repo_path
    )

    if metadata:
        print("\nMetadata:\n")

        for key, value in metadata.items():

            if key == "content":
                print(
                    f"{key}: "
                    f"{len(value)} characters"
                )
            else:
                print(f"{key}: {value}")