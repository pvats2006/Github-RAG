from pathlib import Path
import re

from app.chunking.chunk_schema import CodeChunk


class MarkdownChunker:
    """
    Structure-aware Markdown chunker.

    Markdown is divided using headings while preserving:
    - heading hierarchy
    - section content
    - code blocks
    - line numbers
    """

    def __init__(self, repo_path: str):

        self.repo_path = Path(repo_path)

        self.repo_name = self.repo_path.name

    # --------------------------------------------------
    # READ FILE
    # --------------------------------------------------

    def read_file(self, file_path: Path):

        return file_path.read_text(
            encoding="utf-8",
            errors="ignore"
        )

    # --------------------------------------------------
    # HEADING
    # --------------------------------------------------

    def get_heading_level(self, line: str):

        match = re.match(
            r"^(#{1,6})\s+(.+?)\s*$",
            line
        )

        if not match:
            return None, None

        level = len(
            match.group(1)
        )

        heading = match.group(2).strip()

        return level, heading

    # --------------------------------------------------
    # CHUNK ID
    # --------------------------------------------------

    def make_chunk_id(
        self,
        file_path: Path,
        heading: str,
        line_start: int
    ):

        relative_path = str(
            file_path.relative_to(
                self.repo_path
            )
        )

        safe_heading = re.sub(
            r"[^a-zA-Z0-9_-]+",
            "_",
            heading
        ).strip("_")

        return (
            f"{relative_path}"
            f"::markdown::"
            f"{safe_heading}"
            f"::{line_start}"
        )

    # --------------------------------------------------
    # CREATE CHUNK
    # --------------------------------------------------

    def create_chunk(
        self,
        file_path: Path,
        heading: str,
        level: int,
        line_start: int,
        line_end: int,
        content: str,
        parent_heading=None
    ):

        relative_path = str(
            file_path.relative_to(
                self.repo_path
            )
        )

        chunk_id = self.make_chunk_id(
            file_path,
            heading,
            line_start
        )

        return CodeChunk(

            chunk_id=chunk_id,

            repo=self.repo_name,

            file_path=relative_path,

            file_name=file_path.name,

            language="markdown",

            chunk_type="markdown_section",

            symbol=heading,

            line_start=line_start,

            line_end=line_end,

            content=content,

            parent_symbol=parent_heading,

            metadata={
                "heading_level": level,
                "parent_heading": parent_heading,
            },
        )

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

        lines = source.splitlines()

        chunks = []

        # ----------------------------------------------
        # Heading stack
        #
        # Example:
        #
        # # Project
        # ## Installation
        # ### Requirements
        #
        # Stack keeps track of parent headings.
        # ----------------------------------------------

        heading_stack = []

        current_heading = None
        current_level = None
        current_start = None
        current_parent = None

        # ----------------------------------------------
        # Code block state
        # ----------------------------------------------

        inside_code_block = False

        # ----------------------------------------------
        # Flush current section
        # ----------------------------------------------

        def flush_section(end_line):

            nonlocal current_heading
            nonlocal current_level
            nonlocal current_start
            nonlocal current_parent

            if (
                current_heading is None
                or current_start is None
                or current_level is None
            ):
                return

            if end_line < current_start:
                return

            content = "\n".join(
                lines[
                    current_start - 1:
                    end_line
                ]
            ).strip()

            if not content:
                return

            chunks.append(
                self.create_chunk(
                    file_path=file_path,

                    heading=current_heading,

                    level=current_level,

                    line_start=current_start,

                    line_end=end_line,

                    content=content,

                    parent_heading=current_parent,
                )
            )

        # ----------------------------------------------
        # Process lines
        # ----------------------------------------------

        for index, line in enumerate(
            lines,
            start=1
        ):

            stripped = line.strip()

            # ------------------------------------------
            # Detect fenced code block
            # ------------------------------------------

            if stripped.startswith("```"):

                inside_code_block = (
                    not inside_code_block
                )

                continue

            # ------------------------------------------
            # Ignore headings inside code blocks
            # ------------------------------------------

            if inside_code_block:
                continue

            # ------------------------------------------
            # Detect Markdown heading
            # ------------------------------------------

            level, heading = (
                self.get_heading_level(
                    line
                )
            )

            if level is None:
                continue

            # ------------------------------------------
            # Close previous section
            # ------------------------------------------

            if current_heading is not None:

                flush_section(
                    index - 1
                )

            # ------------------------------------------
            # Remove deeper/equal headings
            # ------------------------------------------

            while (
                heading_stack
                and heading_stack[-1][0]
                >= level
            ):

                heading_stack.pop()

            # ------------------------------------------
            # Parent heading
            # ------------------------------------------

            parent_heading = None

            if heading_stack:

                parent_heading = (
                    heading_stack[-1][1]
                )

            # ------------------------------------------
            # Update heading stack
            # ------------------------------------------

            heading_stack.append(
                (
                    level,
                    heading
                )
            )

            current_heading = heading

            current_level = level

            current_start = index

            current_parent = (
                parent_heading
            )

        # ----------------------------------------------
        # Flush final section
        # ----------------------------------------------

        if current_heading is not None:

            flush_section(
                len(lines)
            )

        # ----------------------------------------------
        # Handle content before first heading
        # ----------------------------------------------

        if chunks:

            first_chunk_start = (
                chunks[0].line_start
            )

            if first_chunk_start > 1:

                preamble_content = "\n".join(
                    lines[
                        0:
                        first_chunk_start - 1
                    ]
                ).strip()

                if preamble_content:

                    relative_path = str(
                        file_path.relative_to(
                            self.repo_path
                        )
                    )

                    preamble = CodeChunk(

                        chunk_id=(
                            f"{relative_path}"
                            f"::markdown::preamble"
                        ),

                        repo=self.repo_name,

                        file_path=relative_path,

                        file_name=file_path.name,

                        language="markdown",

                        chunk_type="markdown_preamble",

                        symbol="preamble",

                        line_start=1,

                        line_end=(
                            first_chunk_start - 1
                        ),

                        content=preamble_content,

                        metadata={
                            "heading_level": 0,
                        },
                    )

                    chunks.insert(
                        0,
                        preamble
                    )

        # ----------------------------------------------
        # Entire file has no headings
        # ----------------------------------------------

        if not chunks:

            relative_path = str(
                file_path.relative_to(
                    self.repo_path
                )
            )

            chunks.append(
                CodeChunk(

                    chunk_id=(
                        f"{relative_path}"
                        f"::markdown::file"
                    ),

                    repo=self.repo_name,

                    file_path=relative_path,

                    file_name=file_path.name,

                    language="markdown",

                    chunk_type="markdown_file",

                    symbol=file_path.name,

                    line_start=1,

                    line_end=len(lines),

                    content=source,

                    metadata={
                        "heading_level": 0,
                    },
                )
            )

        return chunks

    # --------------------------------------------------
    # CHUNK REPOSITORY
    # --------------------------------------------------

    def chunk_repository(self):

        all_chunks = []

        for file_path in self.repo_path.rglob(
            "*.md"
        ):

            # Ignore unwanted directories

            if any(
                part in {
                    ".git",
                    ".venv",
                    "venv",
                    "__pycache__",
                    "node_modules",
                }
                for part in file_path.parts
            ):
                continue

            print(
                f"Chunking Markdown: "
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

    chunker = MarkdownChunker(
        repo_path
    )

    chunks = (
        chunker.chunk_repository()
    )

    print(
        "\nMARKDOWN CHUNKS\n"
    )

    for chunk in chunks:

        print(
            f"{chunk.chunk_type:20} "
            f"{str(chunk.symbol):40} "
            f"{chunk.line_start}-"
            f"{chunk.line_end}"
        )

        if chunk.parent_symbol:

            print(
                f"    parent: "
                f"{chunk.parent_symbol}"
            )

    print(
        f"\nTotal Markdown chunks: "
        f"{len(chunks)}"
    )