from pathlib import Path
from typing import List
import re

from app.chunking.chunk_schema import CodeChunk


class ChunkValidator:
    """
    Validate REAL CodeChunk objects before they enter
    the embedding/vector database pipeline.
    """

    def __init__(self, repo_path: str):

        self.repo_path = Path(repo_path)

    # ==================================================
    # NORMALIZE CONTENT
    # ==================================================

    @staticmethod
    def normalize_content(
        content: str,
        language: str = ""
    ) -> str:

        content = (
            content
            .replace("\r\n", "\n")
            .replace("\r", "\n")
            .strip()
        )

        # ----------------------------------------------
        # HTML / XML
        #
        # HTML parsers may change indentation,
        # line breaks and spaces between tags.
        #
        # We ignore formatting differences but
        # preserve actual tags and content.
        # ----------------------------------------------

        if language in {"html", "xml"}:

            # Remove whitespace between HTML tags.
            #
            # <div>
            #     <p>
            #
            # becomes:
            #
            # <div><p>

            content = re.sub(
                r">\s+<",
                "><",
                content
            )

            # Collapse repeated whitespace.
            #
            # "hello     world"
            # becomes:
            #
            # "hello world"

            content = re.sub(
                r"\s+",
                " ",
                content
            )

            return content.strip()

        # ----------------------------------------------
        # Markdown / text / Python / other formats
        # ----------------------------------------------

        return content

    # ==================================================
    # READ SOURCE
    # ==================================================

    def read_source_lines(
        self,
        file_path: Path
    ):

        return file_path.read_text(
            encoding="utf-8",
            errors="ignore"
        ).splitlines()

    # ==================================================
    # VALIDATE FILE
    # ==================================================

    def validate_file(
        self,
        chunk: CodeChunk
    ):

        errors = []

        absolute_path = (
            self.repo_path
            / chunk.file_path
        )

        if not absolute_path.exists():

            errors.append(
                f"File does not exist: "
                f"{chunk.file_path}"
            )

            return errors, None

        if not absolute_path.is_file():

            errors.append(
                f"Path is not a file: "
                f"{chunk.file_path}"
            )

            return errors, None

        try:

            lines = self.read_source_lines(
                absolute_path
            )

            return errors, lines

        except Exception as exc:

            errors.append(
                f"Could not read file: "
                f"{exc}"
            )

            return errors, None

    # ==================================================
    # VALIDATE LINE RANGE
    # ==================================================

    def validate_line_range(
        self,
        chunk: CodeChunk,
        source_lines: List[str]
    ):

        errors = []

        total_lines = len(
            source_lines
        )

        if chunk.line_start < 1:

            errors.append(
                f"Invalid line_start: "
                f"{chunk.line_start}"
            )

        if chunk.line_end < chunk.line_start:

            errors.append(
                f"Invalid line range: "
                f"{chunk.line_start}-"
                f"{chunk.line_end}"
            )

        if chunk.line_start > total_lines:

            errors.append(
                f"line_start "
                f"{chunk.line_start} "
                f"exceeds file length "
                f"{total_lines}"
            )

        if chunk.line_end > total_lines:

            errors.append(
                f"line_end "
                f"{chunk.line_end} "
                f"exceeds file length "
                f"{total_lines}"
            )

        return errors

    # ==================================================
    # VALIDATE CONTENT AGAINST SOURCE
    # ==================================================

    def validate_content(
        self,
        chunk: CodeChunk,
        source_lines: List[str]
    ):

        errors = []

        # ----------------------------------------------
        # Empty content
        # ----------------------------------------------

        if not chunk.content:

            errors.append(
                "Chunk content is empty"
            )

            return errors

        # ----------------------------------------------
        # Invalid range
        # ----------------------------------------------

        if (
            chunk.line_start < 1
            or chunk.line_end > len(source_lines)
            or chunk.line_end < chunk.line_start
        ):

            return errors

        # ----------------------------------------------
        # Extract actual source
        #
        # Python is zero-indexed.
        # Our line numbers are one-indexed.
        # ----------------------------------------------

        actual_content = "\n".join(
            source_lines[
                chunk.line_start - 1:
                chunk.line_end
            ]
        )

        # ----------------------------------------------
        # Normalize both sides
        # ----------------------------------------------

        actual_normalized = (
            self.normalize_content(
                actual_content,
                chunk.language
            )
        )

        chunk_normalized = (
            self.normalize_content(
                chunk.content,
                chunk.language
            )
        )

        # ----------------------------------------------
        # Exact / normalized match
        # ----------------------------------------------

        if (
            actual_normalized
            == chunk_normalized
        ):

            return errors

        # ----------------------------------------------
        # Fallback:
        # line-by-line comparison
        #
        # Useful for Python / Markdown where
        # line structure should be preserved.
        # ----------------------------------------------

        actual_lines = [
            line.rstrip()
            for line
            in actual_content.splitlines()
        ]

        chunk_lines = [
            line.rstrip()
            for line
            in chunk.content.splitlines()
        ]

        if actual_lines == chunk_lines:

            return errors

        # ----------------------------------------------
        # HTML fallback
        #
        # Some HTML chunkers may remove comments,
        # indentation or parser-generated whitespace.
        #
        # Compare normalized HTML token sequence.
        # ----------------------------------------------

        if chunk.language in {
            "html",
            "xml"
        }:

            actual_html = re.sub(
                r"\s+",
                " ",
                actual_content
            ).strip()

            chunk_html = re.sub(
                r"\s+",
                " ",
                chunk.content
            ).strip()

            if actual_html == chunk_html:

                return errors

        errors.append(
            "Chunk content does not match "
            "the referenced source line range"
        )

        return errors

    # ==================================================
    # VALIDATE SYMBOL RANGE
    # ==================================================

    def validate_symbol(
        self,
        chunk: CodeChunk
    ):

        errors = []

        # File-level chunks don't need symbols.

        if not chunk.symbol:

            return errors

        # Structural formats don't require
        # Python-style symbol validation.

        if chunk.language in {
            "html",
            "json",
            "yaml",
            "text",
            "dockerfile",
            "makefile",
        }:

            return errors

        # Python chunks

        if (
            chunk.chunk_type
            in {
                "function",
                "class",
                "method",
            }
            and not chunk.symbol.strip()
        ):

            errors.append(
                "Python code chunk has empty symbol"
            )

        return errors

    # ==================================================
    # VALIDATE METADATA
    # ==================================================

    def validate_metadata(
        self,
        chunk: CodeChunk
    ):

        errors = []

        if not chunk.chunk_id:

            errors.append(
                "Missing chunk_id"
            )

        if not chunk.file_path:

            errors.append(
                "Missing file_path"
            )

        if not chunk.language:

            errors.append(
                "Missing language"
            )

        if not chunk.chunk_type:

            errors.append(
                "Missing chunk_type"
            )

        # ----------------------------------------------
        # GitHub reference
        # ----------------------------------------------

        reference_available = (
            chunk.metadata.get(
                "github_reference_available"
            )
        )

        reference = (
            chunk.metadata.get(
                "github_reference"
            )
        )

        if (
            reference_available
            and not reference
        ):

            errors.append(
                "GitHub reference is marked "
                "available but URL is missing"
            )

        return errors

    # ==================================================
    # VALIDATE ONE CHUNK
    # ==================================================

    def validate_chunk(
        self,
        chunk: CodeChunk
    ):

        errors = []
        warnings = []

        # ----------------------------------------------
        # Metadata
        # ----------------------------------------------

        errors.extend(
            self.validate_metadata(
                chunk
            )
        )

        # ----------------------------------------------
        # File
        # ----------------------------------------------

        file_errors, source_lines = (
            self.validate_file(
                chunk
            )
        )

        errors.extend(
            file_errors
        )

        if source_lines is None:

            return {
                "valid": False,
                "errors": errors,
                "warnings": warnings,
            }

        # ----------------------------------------------
        # Line range
        # ----------------------------------------------

        errors.extend(
            self.validate_line_range(
                chunk,
                source_lines
            )
        )

        # ----------------------------------------------
        # Content
        # ----------------------------------------------

        if not errors:

            errors.extend(
                self.validate_content(
                    chunk,
                    source_lines
                )
            )

        # ----------------------------------------------
        # Symbol
        # ----------------------------------------------

        errors.extend(
            self.validate_symbol(
                chunk
            )
        )

        # ----------------------------------------------
        # Content size warning
        # ----------------------------------------------

        if len(
            chunk.content.strip()
        ) < 5:

            warnings.append(
                "Chunk content is very small"
            )

        # ----------------------------------------------
        # Relationship metadata
        # ----------------------------------------------

        if chunk.language == "python":

            if (
                not chunk.imports
                and not chunk.calls
                and not chunk.dependencies
            ):

                warnings.append(
                    "Python chunk has no "
                    "relationship metadata"
                )

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
        }

    # ==================================================
    # DUPLICATE CHECK
    # ==================================================

    def find_duplicates(
        self,
        chunks: List[CodeChunk]
    ):

        seen = set()

        duplicates = []

        for index, chunk in enumerate(
            chunks
        ):

            if chunk.chunk_id in seen:

                duplicates.append({
                    "index": index,
                    "chunk_id": chunk.chunk_id,
                })

            else:

                seen.add(
                    chunk.chunk_id
                )

        return duplicates

    # ==================================================
    # VALIDATE ALL CHUNKS
    # ==================================================

    def validate_chunks(
        self,
        chunks: List[CodeChunk]
    ):

        results = []

        valid_chunks = []

        invalid_chunks = []

        # ----------------------------------------------
        # Individual validation
        # ----------------------------------------------

        for index, chunk in enumerate(
            chunks
        ):

            result = (
                self.validate_chunk(
                    chunk
                )
            )

            result["index"] = index

            result["chunk_id"] = (
                chunk.chunk_id
            )

            result["file_path"] = (
                chunk.file_path
            )

            result["symbol"] = (
                chunk.symbol
            )

            results.append(
                result
            )

            if result["valid"]:

                valid_chunks.append(
                    chunk
                )

            else:

                invalid_chunks.append(
                    chunk
                )

        # ----------------------------------------------
        # Duplicates
        # ----------------------------------------------

        duplicates = (
            self.find_duplicates(
                chunks
            )
        )

        # ----------------------------------------------
        # Summary
        # ----------------------------------------------

        summary = {

            "total_chunks":
                len(chunks),

            "valid_chunks":
                len(valid_chunks),

            "invalid_chunks":
                len(invalid_chunks),

            "duplicate_chunks":
                len(duplicates),

            "validation_rate":
                (
                    round(
                        (
                            len(valid_chunks)
                            / len(chunks)
                        ) * 100,
                        2,
                    )
                    if chunks
                    else 0
                ),
        }

        return {

            "summary":
                summary,

            "results":
                results,

            "valid_chunks":
                valid_chunks,

            "invalid_chunks":
                invalid_chunks,

            "duplicates":
                duplicates,
        }

    # ==================================================
    # PRINT REPORT
    # ==================================================

    def print_report(
        self,
        validation_result
    ):

        summary = (
            validation_result[
                "summary"
            ]
        )

        print(
            "\n"
            "========================================"
        )

        print(
            "CHUNK VALIDATION REPORT"
        )

        print(
            "========================================"
        )

        print(
            f"Total chunks:      "
            f"{summary['total_chunks']}"
        )

        print(
            f"Valid chunks:      "
            f"{summary['valid_chunks']}"
        )

        print(
            f"Invalid chunks:    "
            f"{summary['invalid_chunks']}"
        )

        print(
            f"Duplicate chunks:  "
            f"{summary['duplicate_chunks']}"
        )

        print(
            f"Validation rate:   "
            f"{summary['validation_rate']}%"
        )

        # ----------------------------------------------
        # Invalid chunks
        # ----------------------------------------------

        results = (
            validation_result[
                "results"
            ]
        )

        invalid_found = False

        print(
            "\nINVALID CHUNKS\n"
        )

        for result in results:

            if result["valid"]:

                continue

            invalid_found = True

            print(
                f"❌ {result['file_path']}"
                f" :: "
                f"{result['symbol']}"
            )

            for error in result[
                "errors"
            ]:

                print(
                    f"   → {error}"
                )

        if not invalid_found:

            print(
                "None 🎉"
            )

        # ----------------------------------------------
        # Duplicates
        # ----------------------------------------------

        print(
            "\nDUPLICATES\n"
        )

        duplicates = (
            validation_result[
                "duplicates"
            ]
        )

        if duplicates:

            for duplicate in duplicates:

                print(
                    f"❌ "
                    f"{duplicate['chunk_id']}"
                )

        else:

            print(
                "None 🎉"
            )