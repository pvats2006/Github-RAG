import json
from pathlib import Path
from typing import List, Dict, Any

from app.chunking.chunk_schema import CodeChunk


class RelationshipEnricher:
    """
    Enrich CodeChunk objects using repository_knowledge.json.

    Adds:
        imports
        calls
        dependencies
        related_files
        relationship metadata
    """

    def __init__(
        self,
        knowledge_path: str
    ):

        self.knowledge_path = Path(
            knowledge_path
        )

        with open(
            self.knowledge_path,
            "r",
            encoding="utf-8"
        ) as f:

            self.knowledge = json.load(f)

        self.files = self.knowledge.get(
            "files",
            []
        )

        # file_path -> file information
        self.file_index = {
            file_info["path"]: file_info
            for file_info in self.files
        }

        # symbol -> locations
        self.symbol_index = {}

        self._build_symbol_index()

    # ==================================================
    # SYMBOL INDEX
    # ==================================================

    def _build_symbol_index(self):

        for file_info in self.files:

            file_path = file_info.get(
                "path"
            )

            for symbol in file_info.get(
                "symbols",
                []
            ):

                name = symbol.get(
                    "name"
                )

                if not name:
                    continue

                entry = {
                    "name": name,
                    "type": symbol.get(
                        "type"
                    ),
                    "file": file_path,
                    "line_start": symbol.get(
                        "line_start"
                    ),
                    "line_end": symbol.get(
                        "line_end"
                    ),
                }

                self.symbol_index.setdefault(
                    name,
                    []
                ).append(entry)

    # ==================================================
    # FILE INFO
    # ==================================================

    def get_file_info(
        self,
        file_path: str
    ):

        return self.file_index.get(
            file_path
        )

    # ==================================================
    # FILE IMPORTS
    # ==================================================

    def get_file_imports(
        self,
        file_info: Dict[str, Any]
    ):

        return file_info.get(
            "imports",
            []
        )

    # ==================================================
    # FILE RELATIONSHIPS
    # ==================================================

    def get_file_relationships(
        self,
        file_info: Dict[str, Any]
    ):

        return file_info.get(
            "relationships",
            []
        )

    # ==================================================
    # NORMALIZE SYMBOL
    # ==================================================

    def normalize_symbol(
        self,
        symbol: str
    ):

        if not symbol:
            return ""

        return (
            symbol
            .replace(
                "function:",
                ""
            )
            .replace(
                "method:",
                ""
            )
            .replace(
                "class:",
                ""
            )
            .strip()
        )

    # ==================================================
    # CHECK SOURCE MATCH
    # ==================================================

    def source_matches_chunk(
        self,
        source: str,
        chunk: CodeChunk
    ):

        if not source or not chunk.symbol:
            return False

        normalized_source = (
            self.normalize_symbol(
                source
            )
        )

        chunk_symbol = (
            self.normalize_symbol(
                chunk.symbol
            )
        )

        # ----------------------------------------------
        # Exact
        # ----------------------------------------------

        if (
            normalized_source
            == chunk_symbol
        ):
            return True

        # ----------------------------------------------
        # Ends with class/method
        #
        # Example:
        #
        # src/file.py:method:ModelTrainer.foo
        # ----------------------------------------------

        if normalized_source.endswith(
            f":{chunk_symbol}"
        ):

            return True

        # ----------------------------------------------
        # Last component
        # ----------------------------------------------

        source_last = (
            normalized_source
            .split(":")[-1]
        )

        chunk_last = (
            chunk_symbol
            .split(".")[-1]
        )

        if source_last == chunk_last:
            return True

        return False

    # ==================================================
    # EXTRACT CALL NAME
    # ==================================================

    def extract_call_name(
        self,
        target: str
    ):

        if not target:
            return None

        # Example:
        #
        # application.py:call:app.route:13
        #
        # src/file.py:call:evaluate_models:55
        #

        parts = target.split(":")

        if len(parts) >= 3:

            if parts[1] == "call":

                return parts[2]

        return target

    # ==================================================
    # GET SYMBOL CALLS
    # ==================================================

    def get_symbol_calls(
        self,
        file_info: Dict[str, Any],
        chunk: CodeChunk
    ):

        calls = []

        relationships = (
            self.get_file_relationships(
                file_info
            )
        )

        for relationship in relationships:

            if relationship.get(
                "relation"
            ) != "calls":

                continue

            source = relationship.get(
                "source",
                ""
            )

            if not self.source_matches_chunk(
                source,
                chunk
            ):

                continue

            target = relationship.get(
                "target",
                ""
            )

            call_name = (
                self.extract_call_name(
                    target
                )
            )

            if call_name:

                calls.append(
                    call_name
                )

        return list(
            dict.fromkeys(
                calls
            )
        )

    # ==================================================
    # GET IMPORTS + DEPENDENCIES
    # ==================================================

    def get_import_dependencies(
        self,
        file_info: Dict[str, Any]
    ):

        imports = []

        dependencies = []

        relationship_targets = []

        # ----------------------------------------------
        # Import records
        # ----------------------------------------------

        for import_info in (
            self.get_file_imports(
                file_info
            )
        ):

            target_file = (
                import_info.get(
                    "target_file"
                )
            )

            if target_file:

                dependencies.append(
                    target_file
                )

            symbols = (
                import_info.get(
                    "symbols",
                    []
                )
            )

            imports.extend(
                symbols
            )

        # ----------------------------------------------
        # Import relationships
        # ----------------------------------------------

        for relationship in (
            self.get_file_relationships(
                file_info
            )
        ):

            if relationship.get(
                "relation"
            ) != "imports":

                continue

            target = relationship.get(
                "target"
            )

            if target:

                relationship_targets.append(
                    target
                )

        return (
            list(
                dict.fromkeys(
                    imports
                )
            ),
            list(
                dict.fromkeys(
                    dependencies
                )
            ),
            list(
                dict.fromkeys(
                    relationship_targets
                )
            ),
        )

    # ==================================================
    # FIND RELATED FILES
    # ==================================================

    def get_related_files(
        self,
        file_path: str
    ):

        related_files = []

        file_info = self.get_file_info(
            file_path
        )

        if not file_info:
            return []

        # ----------------------------------------------
        # Outgoing dependencies
        # ----------------------------------------------

        for import_info in file_info.get(
            "imports",
            []
        ):

            target = import_info.get(
                "target_file"
            )

            if target:

                related_files.append(
                    target
                )

        # ----------------------------------------------
        # Reverse dependencies
        # ----------------------------------------------

        for other_file in self.files:

            other_path = other_file.get(
                "path"
            )

            if other_path == file_path:
                continue

            for import_info in other_file.get(
                "imports",
                []
            ):

                target = import_info.get(
                    "target_file"
                )

                if target == file_path:

                    if other_path:

                        related_files.append(
                            other_path
                        )

        return list(
            dict.fromkeys(
                related_files
            )
        )

    # ==================================================
    # ENRICH ONE CHUNK
    # ==================================================

    def enrich_chunk(
        self,
        chunk: CodeChunk
    ):

        file_info = self.get_file_info(
            chunk.file_path
        )

        # ----------------------------------------------
        # No repository information
        # ----------------------------------------------

        if not file_info:

            chunk.metadata.update({
                "relationship_enriched": False,
                "relationship_reason":
                    "file_not_found_in_knowledge",
            })

            return chunk

        # ----------------------------------------------
        # Imports / dependencies
        # ----------------------------------------------

        (
            imports,
            dependencies,
            relationship_targets
        ) = self.get_import_dependencies(
            file_info
        )

        # ----------------------------------------------
        # Calls
        # ----------------------------------------------

        calls = []

        if chunk.symbol:

            calls = self.get_symbol_calls(
                file_info,
                chunk
            )

        # ----------------------------------------------
        # Related files
        # ----------------------------------------------

        related_files = (
            self.get_related_files(
                chunk.file_path
            )
        )

        # ----------------------------------------------
        # Assign
        # ----------------------------------------------

        chunk.imports = imports

        chunk.calls = calls

        chunk.dependencies = (
            dependencies
        )

        chunk.related_files = (
            related_files
        )

        # ----------------------------------------------
        # Metadata
        # ----------------------------------------------

        chunk.metadata.update({

            "relationship_enriched": True,

            "relationship_targets":
                relationship_targets,

            "repository":
                self.knowledge.get(
                    "repository"
                ),

            "dependency_count":
                len(dependencies),

            "related_file_count":
                len(related_files),

            "call_count":
                len(calls),

        })

        return chunk

    # ==================================================
    # ENRICH MANY CHUNKS
    # ==================================================

    def enrich_chunks(
        self,
        chunks: List[CodeChunk]
    ):

        enriched = []

        for chunk in chunks:

            enriched.append(
                self.enrich_chunk(
                    chunk
                )
            )

        return enriched


# ======================================================
# TEST
# ======================================================

if __name__ == "__main__":

    knowledge_path = input(
        "Enter repository knowledge path: "
    ).strip()

    file_path = input(
        "Enter file path to inspect: "
    ).strip()

    chunk_type = input(
        "Enter chunk type "
        "(function/class/method/file): "
    ).strip()

    symbol = input(
        "Enter symbol "
        "(example: ModelTrainer.initiate_model_trainer): "
    ).strip()

    chunk = CodeChunk(

        chunk_id="test",

        repo="ML-Project",

        file_path=file_path,

        file_name=Path(
            file_path
        ).name,

        language="python",

        chunk_type=chunk_type,

        symbol=symbol,

        line_start=1,

        line_end=1,

        content="",
    )

    enricher = RelationshipEnricher(
        knowledge_path
    )

    chunk = enricher.enrich_chunk(
        chunk
    )

    print(
        "\nENRICHED CHUNK\n"
    )

    print(
        "File:",
        chunk.file_path
    )

    print(
        "Symbol:",
        chunk.symbol
    )

    print(
        "Imports:",
        chunk.imports
    )

    print(
        "Calls:",
        chunk.calls
    )

    print(
        "Dependencies:",
        chunk.dependencies
    )

    print(
        "Related files:",
        chunk.related_files
    )

    print(
        "Metadata:",
        chunk.metadata
    )