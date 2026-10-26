import json
from dataclasses import asdict
from pathlib import Path

from app.chunking.python_chunker import PythonChunker
from app.chunking.markdown_chunker import MarkdownChunker
from app.chunking.html_chunker import HTMLChunker
from app.chunking.config_chunker import ConfigChunker

from app.chunking.relationship_enricher import (
    RelationshipEnricher
)

from app.chunking.github_reference import (
    GitHubReferenceBuilder
)

from app.chunking.chunk_validator import (
    ChunkValidator
)


class ChunkPipeline:
    """
    Complete Phase 3 chunking pipeline.

    1. Generate chunks
    2. Enrich relationships
    3. Add GitHub references
    4. Validate chunks
    5. Save final chunks
    """

    def __init__(
        self,
        repo_path: str,
        knowledge_path: str
    ):

        self.repo_path = Path(
            repo_path
        )

        self.knowledge_path = Path(
            knowledge_path
        )

    # ==================================================
    # GENERATE CHUNKS
    # ==================================================

    def generate_chunks(self):

        chunks = []

        # ----------------------------------------------
        # Python
        # ----------------------------------------------

        print(
            "\n[1/4] Python chunking..."
        )

        python_chunker = (
            PythonChunker(
                str(self.repo_path)
            )
        )

        python_chunks = (
            python_chunker.chunk_repository()
        )

        print(
            f"Python chunks: "
            f"{len(python_chunks)}"
        )

        chunks.extend(
            python_chunks
        )

        # ----------------------------------------------
        # Markdown
        # ----------------------------------------------

        print(
            "\n[2/4] Markdown chunking..."
        )

        markdown_chunker = (
            MarkdownChunker(
                str(self.repo_path)
            )
        )

        markdown_chunks = (
            markdown_chunker.chunk_repository()
        )

        print(
            f"Markdown chunks: "
            f"{len(markdown_chunks)}"
        )

        chunks.extend(
            markdown_chunks
        )

        # ----------------------------------------------
        # HTML
        # ----------------------------------------------

        print(
            "\n[3/4] HTML chunking..."
        )

        html_chunker = (
            HTMLChunker(
                str(self.repo_path)
            )
        )

        html_chunks = (
            html_chunker.chunk_repository()
        )

        print(
            f"HTML chunks: "
            f"{len(html_chunks)}"
        )

        chunks.extend(
            html_chunks
        )

        # ----------------------------------------------
        # Config/Text
        # ----------------------------------------------

        print(
            "\n[4/4] Config/Text chunking..."
        )

        config_chunker = (
            ConfigChunker(
                str(self.repo_path)
            )
        )

        config_chunks = (
            config_chunker.chunk_repository()
        )

        print(
            f"Config/Text chunks: "
            f"{len(config_chunks)}"
        )

        chunks.extend(
            config_chunks
        )

        return chunks

    # ==================================================
    # ENRICH CHUNKS
    # ==================================================

    def enrich_chunks(
        self,
        chunks
    ):

        print(
            "\nEnriching relationships..."
        )

        relationship_enricher = (
            RelationshipEnricher(
                str(self.knowledge_path)
            )
        )

        chunks = (
            relationship_enricher.enrich_chunks(
                chunks
            )
        )

        print(
            "Relationships enriched."
        )

        print(
            "\nAdding GitHub references..."
        )

        github_builder = (
            GitHubReferenceBuilder(
                str(self.repo_path)
            )
        )

        chunks = (
            github_builder.build_references(
                chunks
            )
        )

        print(
            "GitHub references added."
        )

        return chunks

    # ==================================================
    # VALIDATE
    # ==================================================

    def validate(
        self,
        chunks
    ):

        print(
            "\nValidating REAL chunks..."
        )

        validator = ChunkValidator(
            str(self.repo_path)
        )

        result = (
            validator.validate_chunks(
                chunks
            )
        )

        validator.print_report(
            result
        )

        return result

    # ==================================================
    # SAVE
    # ==================================================

    def save_chunks(
        self,
        chunks,
        output_path: str
    ):

        output = Path(
            output_path
        )

        output.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        data = []

        for chunk in chunks:

            data.append(
                asdict(chunk)
            )

        with open(
            output,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                data,
                f,
                indent=2,
                ensure_ascii=False
            )

        print(
            f"\nFinal chunks saved:"
            f"\n{output}"
        )

    # ==================================================
    # RUN
    # ==================================================

    def run(self):

        print(
            "\n========================================"
        )

        print(
            "REPOSITORY CHUNK PIPELINE"
        )

        print(
            "========================================"
        )

        # ----------------------------------------------
        # 1. Generate
        # ----------------------------------------------

        chunks = (
            self.generate_chunks()
        )

        print(
            f"\nTotal generated chunks: "
            f"{len(chunks)}"
        )

        # ----------------------------------------------
        # 2. Enrich
        # ----------------------------------------------

        chunks = (
            self.enrich_chunks(
                chunks
            )
        )

        # ----------------------------------------------
        # 3. Validate
        # ----------------------------------------------

        validation = (
            self.validate(
                chunks
            )
        )

        # ----------------------------------------------
        # 4. Save ONLY valid chunks
        # ----------------------------------------------

        valid_chunks = (
            validation[
                "valid_chunks"
            ]
        )

        self.save_chunks(
            valid_chunks,
            "data/chunks/final_chunks.json"
        )

        print(
            "\n========================================"
        )

        print(
            "PIPELINE COMPLETE"
        )

        print(
            "========================================"
        )

        print(
            f"Generated: "
            f"{len(chunks)}"
        )

        print(
            f"Valid: "
            f"{len(valid_chunks)}"
        )

        print(
            f"Rejected: "
            f"{len(validation['invalid_chunks'])}"
        )

        return {
            "chunks": valid_chunks,
            "validation": validation,
        }


# ======================================================
# MAIN
# ======================================================

if __name__ == "__main__":

    repo_path = input(
        "Enter repository path: "
    ).strip()

    knowledge_path = input(
        "Enter repository knowledge path: "
    ).strip()

    pipeline = ChunkPipeline(
        repo_path=repo_path,
        knowledge_path=knowledge_path,
    )

    pipeline.run()