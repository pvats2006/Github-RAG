import json
import re
from pathlib import Path

from app.chunking.chunk_schema import CodeChunk


IGNORED_DIRS = {
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    "node_modules",
    "artifacts",
    "catboost_info",
}


SUPPORTED_EXTENSIONS = {
    ".json",
    ".yaml",
    ".yml",
    ".txt",
}


SPECIAL_FILES = {
    "Dockerfile",
    "Makefile",
    ".env",
    ".env.example",
    ".gitignore",
}


class ConfigChunker:
    """
    Structure-aware chunker for configuration,
    JSON, YAML and text files.
    """

    MAX_LINES = 120

    def __init__(
        self,
        repo_path: str
    ):

        self.repo_path = Path(
            repo_path
        )

        self.repo_name = (
            self.repo_path.name
        )

    # --------------------------------------------------
    # READ FILE
    # --------------------------------------------------

    def read_file(
        self,
        file_path: Path
    ):

        return file_path.read_text(
            encoding="utf-8",
            errors="ignore"
        )

    # --------------------------------------------------
    # LANGUAGE
    # --------------------------------------------------

    def get_language(
        self,
        file_path: Path
    ):

        name = file_path.name.lower()

        if name == "dockerfile":
            return "dockerfile"

        if name == "makefile":
            return "makefile"

        extension = (
            file_path.suffix.lower()
        )

        mapping = {
            ".json": "json",
            ".yaml": "yaml",
            ".yml": "yaml",
            ".txt": "text",
        }

        return mapping.get(
            extension,
            "text"
        )

    # --------------------------------------------------
    # CHUNK ID
    # --------------------------------------------------

    def make_chunk_id(
        self,
        file_path: Path,
        symbol: str,
        line_start: int
    ):

        relative_path = str(
            file_path.relative_to(
                self.repo_path
            )
        )

        safe_symbol = re.sub(
            r"[^a-zA-Z0-9_-]+",
            "_",
            symbol
        ).strip("_")

        return (
            f"{relative_path}"
            f"::config::"
            f"{safe_symbol}"
            f"::{line_start}"
        )

    # --------------------------------------------------
    # CREATE CHUNK
    # --------------------------------------------------

    def create_chunk(
        self,
        file_path: Path,
        chunk_type: str,
        symbol: str,
        line_start: int,
        line_end: int,
        content: str,
        metadata=None,
    ):

        relative_path = str(
            file_path.relative_to(
                self.repo_path
            )
        )

        chunk_id = self.make_chunk_id(
            file_path,
            symbol,
            line_start
        )

        return CodeChunk(

            chunk_id=chunk_id,

            repo=self.repo_name,

            file_path=relative_path,

            file_name=file_path.name,

            language=self.get_language(
                file_path
            ),

            chunk_type=chunk_type,

            symbol=symbol,

            line_start=line_start,

            line_end=line_end,

            content=content,

            metadata=metadata or {},
        )

    # --------------------------------------------------
    # JSON CHUNKING
    # --------------------------------------------------

    def chunk_json(
        self,
        file_path: Path,
        source: str
    ):

        try:

            data = json.loads(
                source
            )

        except json.JSONDecodeError:

            return self.chunk_text(
                file_path,
                source
            )

        lines = source.splitlines()

        # ----------------------------------------------
        # Small JSON → entire file
        # ----------------------------------------------

        if len(lines) <= self.MAX_LINES:

            return [
                self.create_chunk(
                    file_path=file_path,

                    chunk_type="json_file",

                    symbol=file_path.name,

                    line_start=1,

                    line_end=len(lines),

                    content=source,

                    metadata={
                        "format": "json"
                    },
                )
            ]

        chunks = []

        # ----------------------------------------------
        # Large JSON dictionary
        # ----------------------------------------------

        if isinstance(
            data,
            dict
        ):

            for key, value in data.items():

                content = json.dumps(
                    {
                        key: value
                    },
                    indent=2,
                    ensure_ascii=False,
                )

                chunks.append(
                    self.create_chunk(
                        file_path=file_path,

                        chunk_type="json_section",

                        symbol=str(key),

                        line_start=1,

                        line_end=len(lines),

                        content=content,

                        metadata={
                            "format": "json",
                            "json_key": str(key),
                        },
                    )
                )

            return chunks

        # ----------------------------------------------
        # Large JSON list
        # ----------------------------------------------

        return [
            self.create_chunk(
                file_path=file_path,

                chunk_type="json_file",

                symbol=file_path.name,

                line_start=1,

                line_end=len(lines),

                content=source,

                metadata={
                    "format": "json"
                },
            )
        ]

    # --------------------------------------------------
    # YAML CHUNKING
    # --------------------------------------------------

    def chunk_yaml(
        self,
        file_path: Path,
        source: str
    ):

        lines = source.splitlines()

        if not lines:
            return []

        chunks = []

        current_key = None

        current_start = None

        current_lines = []

        for index, line in enumerate(
            lines,
            start=1
        ):

            stripped = line.strip()

            # Ignore blank lines/comments
            if (
                not stripped
                or stripped.startswith("#")
            ):

                if current_key is not None:

                    current_lines.append(
                        line
                    )

                continue

            # ------------------------------------------
            # Detect top-level YAML key
            # ------------------------------------------

            if (
                not line.startswith(
                    (" ", "\t")
                )
                and ":" in line
            ):

                # Flush previous section

                if (
                    current_key is not None
                    and current_start is not None
                ):

                    content = "\n".join(
                        current_lines
                    ).strip()

                    if content:

                        chunks.append(
                            self.create_chunk(
                                file_path=file_path,

                                chunk_type="yaml_section",

                                symbol=current_key,

                                line_start=current_start,

                                line_end=index - 1,

                                content=content,

                                metadata={
                                    "format": "yaml",
                                    "yaml_key": current_key,
                                },
                            )
                        )

                current_key = (
                    stripped.split(
                        ":",
                        1
                    )[0].strip()
                )

                current_start = index

                current_lines = [
                    line
                ]

            else:

                if current_key is not None:

                    current_lines.append(
                        line
                    )

        # ----------------------------------------------
        # Flush final section
        # ----------------------------------------------

        if (
            current_key is not None
            and current_start is not None
        ):

            content = "\n".join(
                current_lines
            ).strip()

            if content:

                chunks.append(
                    self.create_chunk(
                        file_path=file_path,

                        chunk_type="yaml_section",

                        symbol=current_key,

                        line_start=current_start,

                        line_end=len(lines),

                        content=content,

                        metadata={
                            "format": "yaml",
                            "yaml_key": current_key,
                        },
                    )
                )

        if chunks:

            return chunks

        return self.chunk_text(
            file_path,
            source
        )

    # --------------------------------------------------
    # TEXT CHUNKING
    # --------------------------------------------------

    def chunk_text(
        self,
        file_path: Path,
        source: str
    ):

        lines = source.splitlines()

        if not lines:
            return []

        # ----------------------------------------------
        # Small file → one chunk
        # ----------------------------------------------

        if len(lines) <= self.MAX_LINES:

            return [
                self.create_chunk(
                    file_path=file_path,

                    chunk_type="text_file",

                    symbol=file_path.name,

                    line_start=1,

                    line_end=len(lines),

                    content=source,

                    metadata={
                        "format": "text"
                    },
                )
            ]

        # ----------------------------------------------
        # Large text → blocks
        # ----------------------------------------------

        chunks = []

        for start in range(
            0,
            len(lines),
            self.MAX_LINES
        ):

            end = min(
                start + self.MAX_LINES,
                len(lines)
            )

            content = "\n".join(
                lines[start:end]
            ).strip()

            if not content:
                continue

            block_number = (
                start // self.MAX_LINES
            ) + 1

            chunks.append(
                self.create_chunk(
                    file_path=file_path,

                    chunk_type="text_block",

                    symbol=(
                        f"{file_path.name}"
                        f"::block_{block_number}"
                    ),

                    line_start=start + 1,

                    line_end=end,

                    content=content,

                    metadata={
                        "format": "text",
                        "block_number": block_number,
                    },
                )
            )

        return chunks

    # --------------------------------------------------
    # CHUNK FILE
    # --------------------------------------------------

    def chunk_file(
        self,
        file_path: Path
    ):

        source = self.read_file(
            file_path
        )

        if not source.strip():
            return []

        extension = (
            file_path.suffix.lower()
        )

        name = (
            file_path.name
        )

        if extension == ".json":

            return self.chunk_json(
                file_path,
                source
            )

        if extension in {
            ".yaml",
            ".yml",
        }:

            return self.chunk_yaml(
                file_path,
                source
            )

        return self.chunk_text(
            file_path,
            source
        )

    # --------------------------------------------------
    # CHUNK REPOSITORY
    # --------------------------------------------------

    def chunk_repository(self):

        all_chunks = []

        for file_path in self.repo_path.rglob(
            "*"
        ):

            if not file_path.is_file():
                continue

            if any(
                part in IGNORED_DIRS
                for part in file_path.parts
            ):
                continue

            if (
                file_path.suffix.lower()
                not in SUPPORTED_EXTENSIONS
                and file_path.name
                not in SPECIAL_FILES
            ):
                continue

            print(
                f"Chunking Config: "
                f"{file_path}"
            )

            chunks = self.chunk_file(
                file_path
            )

            all_chunks.extend(
                chunks
            )

        return all_chunks


# ======================================================
# TEST
# ======================================================

if __name__ == "__main__":

    repo_path = input(
        "Enter repository path: "
    ).strip()

    chunker = ConfigChunker(
        repo_path
    )

    chunks = (
        chunker.chunk_repository()
    )

    print(
        "\nCONFIG / TEXT CHUNKS\n"
    )

    for chunk in chunks:

        print(
            f"{chunk.chunk_type:18} "
            f"{str(chunk.symbol):40} "
            f"{chunk.line_start}-"
            f"{chunk.line_end}"
        )

    print(
        f"\nTotal config/text chunks: "
        f"{len(chunks)}"
    )