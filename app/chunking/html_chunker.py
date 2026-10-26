from pathlib import Path
import re
from html.parser import HTMLParser

from app.chunking.chunk_schema import CodeChunk


IGNORED_DIRS = {
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    "node_modules",
}


class HTMLSectionParser(HTMLParser):
    """
    HTML structure parser.

    Important:
    This parser ONLY finds section boundaries.
    Actual chunk content is taken directly from
    the original source file.
    """

    TRACKED_TAGS = {
        "html",
        "head",
        "body",
        "header",
        "nav",
        "main",
        "section",
        "article",
        "aside",
        "footer",
        "form",
        "script",
        "style",
    }

    def __init__(self):

        super().__init__(
            convert_charrefs=False
        )

        self.stack = []
        self.sections = []

    # ==================================================
    # START TAG
    # ==================================================

    def handle_starttag(
        self,
        tag,
        attrs
    ):

        tag = tag.lower()

        if tag not in self.TRACKED_TAGS:
            return

        line = self.getpos()[0]

        self.stack.append(
            {
                "tag": tag,
                "line_start": line,
                "line_end": None,
            }
        )

    # ==================================================
    # END TAG
    # ==================================================

    def handle_endtag(
        self,
        tag
    ):

        tag = tag.lower()

        if tag not in self.TRACKED_TAGS:
            return

        end_line = self.getpos()[0]

        # Find the latest matching opening tag.
        for index in range(
            len(self.stack) - 1,
            -1,
            -1
        ):

            section = self.stack[index]

            if section["tag"] == tag:

                section["line_end"] = end_line

                self.sections.append(
                    section
                )

                del self.stack[index]

                break


class HTMLChunker:
    """
    Structure-aware HTML chunker.

    Creates chunks for meaningful HTML
    structures such as:

        html
        head
        body
        form
        script
        style
        section
        article
        nav

    Content is ALWAYS extracted from the
    original source lines.
    """

    MIN_SECTION_LINES = 2

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

    # ==================================================
    # READ FILE
    # ==================================================

    def read_file(
        self,
        file_path: Path
    ):

        return file_path.read_text(
            encoding="utf-8",
            errors="ignore"
        )

    # ==================================================
    # CHUNK ID
    # ==================================================

    def make_chunk_id(
        self,
        file_path: Path,
        section: str,
        line_start: int
    ):

        relative_path = str(
            file_path.relative_to(
                self.repo_path
            )
        )

        safe_section = re.sub(
            r"[^a-zA-Z0-9_-]+",
            "_",
            section
        ).strip("_")

        return (
            f"{relative_path}"
            f"::html::"
            f"{safe_section}"
            f"::{line_start}"
        )

    # ==================================================
    # CREATE CHUNK
    # ==================================================

    def create_chunk(
        self,
        file_path: Path,
        section: str,
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

        return CodeChunk(

            chunk_id=self.make_chunk_id(
                file_path,
                section,
                line_start
            ),

            repo=self.repo_name,

            file_path=relative_path,

            file_name=file_path.name,

            language="html",

            chunk_type="html_section",

            symbol=section,

            line_start=line_start,

            line_end=line_end,

            content=content,

            metadata=metadata or {},
        )

    # ==================================================
    # FILE CHUNK
    # ==================================================

    def create_file_chunk(
        self,
        file_path: Path,
        source: str
    ):

        lines = source.splitlines()

        relative_path = str(
            file_path.relative_to(
                self.repo_path
            )
        )

        return CodeChunk(

            chunk_id=(
                f"{relative_path}"
                f"::html::file"
            ),

            repo=self.repo_name,

            file_path=relative_path,

            file_name=file_path.name,

            language="html",

            chunk_type="html_file",

            symbol=file_path.name,

            line_start=1,

            line_end=len(lines),

            content=source,

            metadata={
                "html_tag": "file",
                "line_count": len(lines),
            },
        )

    # ==================================================
    # EXTRACT ORIGINAL SOURCE LINES
    # ==================================================

    def extract_lines(
        self,
        source_lines,
        line_start,
        line_end
    ):
        """
        Extract exact original source lines.

        HTMLParser gives us 1-based line numbers.
        Python lists are 0-based.
        """

        return "\n".join(
            source_lines[
                line_start - 1:
                line_end
            ]
        )

    # ==================================================
    # CHUNK FILE
    # ==================================================

    def chunk_file(
        self,
        file_path: Path
    ):

        source = self.read_file(
            file_path
        )

        if not source.strip():

            return []

        source_lines = (
            source.splitlines()
        )

        parser = HTMLSectionParser()

        try:

            parser.feed(source)
            parser.close()

        except Exception as exc:

            print(
                f"HTML parsing warning: "
                f"{file_path}: {exc}"
            )

        sections = parser.sections

        # ------------------------------------------------
        # No structural sections
        # ------------------------------------------------

        if not sections:

            return [
                self.create_file_chunk(
                    file_path,
                    source
                )
            ]

        chunks = []

        # ------------------------------------------------
        # Sort sections
        # ------------------------------------------------

        sections.sort(
            key=lambda item: (
                item["line_start"],
                item["line_end"] or 0
            )
        )

        # ------------------------------------------------
        # Create chunks
        # ------------------------------------------------

        seen = set()

        for section in sections:

            tag = section["tag"]

            start_line = (
                section["line_start"]
            )

            end_line = (
                section["line_end"]
                or start_line
            )

            # --------------------------------------------
            # Validate parser boundaries
            # --------------------------------------------

            if start_line < 1:

                continue

            if end_line < start_line:

                continue

            if start_line > len(
                source_lines
            ):

                continue

            end_line = min(
                end_line,
                len(source_lines)
            )

            # --------------------------------------------
            # Section size
            # --------------------------------------------

            line_count = (
                end_line
                - start_line
                + 1
            )

            if (
                line_count
                < self.MIN_SECTION_LINES
            ):

                continue

            # --------------------------------------------
            # Exact original source content
            # --------------------------------------------

            content = self.extract_lines(
                source_lines,
                start_line,
                end_line
            )

            if not content.strip():

                continue

            # --------------------------------------------
            # Duplicate protection
            # --------------------------------------------

            chunk_key = (
                tag,
                start_line,
                end_line
            )

            if chunk_key in seen:

                continue

            seen.add(
                chunk_key
            )

            # --------------------------------------------
            # Create chunk
            # --------------------------------------------

            chunks.append(
                self.create_chunk(

                    file_path=file_path,

                    section=tag,

                    line_start=start_line,

                    line_end=end_line,

                    content=content,

                    metadata={
                        "html_tag": tag,
                        "line_count": line_count,
                        "content_source": (
                            "original_file_lines"
                        ),
                    },
                )
            )

        # ------------------------------------------------
        # Fallback
        # ------------------------------------------------

        if not chunks:

            return [
                self.create_file_chunk(
                    file_path,
                    source
                )
            ]

        return chunks

    # ==================================================
    # CHUNK REPOSITORY
    # ==================================================

    def chunk_repository(self):

        all_chunks = []

        for file_path in self.repo_path.rglob(
            "*.html"
        ):

            # Ignore unwanted directories

            if any(
                part in IGNORED_DIRS
                for part in file_path.parts
            ):

                continue

            print(
                f"Chunking HTML: "
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

    chunker = HTMLChunker(
        repo_path
    )

    chunks = (
        chunker.chunk_repository()
    )

    print(
        "\nHTML CHUNKS\n"
    )

    for chunk in chunks:

        print(
            f"{chunk.chunk_type:18} "
            f"{str(chunk.symbol):20} "
            f"{chunk.line_start}-"
            f"{chunk.line_end}"
        )

    print(
        f"\nTotal HTML chunks: "
        f"{len(chunks)}"
    )