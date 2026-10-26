import json
from pathlib import Path
from typing import List, Dict, Optional

from app.chunking.chunk_schema import CodeChunk


class ChunkStore:
    """
    Persistent storage and retrieval layer for
    validated repository chunks.

    This layer sits between:

        Chunking Pipeline
              ↓
        Final Chunk Store
              ↓
        Embedding Pipeline
    """

    def __init__(
        self,
        storage_path: str
    ):

        self.storage_path = Path(
            storage_path
        )

        self.chunks: List[CodeChunk] = []

        self.chunk_index = {}

    # ==================================================
    # LOAD
    # ==================================================

    def load(self):

        if not self.storage_path.exists():

            raise FileNotFoundError(
                f"Chunk store not found: "
                f"{self.storage_path}"
            )

        with open(
            self.storage_path,
            "r",
            encoding="utf-8"
        ) as f:

            data = json.load(f)

        self.chunks = [
            CodeChunk(
                **item
            )
            for item in data
        ]

        self.build_index()

        return self.chunks

    # ==================================================
    # BUILD INDEX
    # ==================================================

    def build_index(self):

        self.chunk_index = {}

        for chunk in self.chunks:

            self.chunk_index[
                chunk.chunk_id
            ] = chunk

    # ==================================================
    # GET BY ID
    # ==================================================

    def get_by_id(
        self,
        chunk_id: str
    ) -> Optional[CodeChunk]:

        return self.chunk_index.get(
            chunk_id
        )

    # ==================================================
    # GET BY FILE
    # ==================================================

    def get_by_file(
        self,
        file_path: str
    ) -> List[CodeChunk]:

        return [
            chunk
            for chunk in self.chunks
            if chunk.file_path == file_path
        ]

    # ==================================================
    # GET BY SYMBOL
    # ==================================================

    def get_by_symbol(
        self,
        symbol: str
    ) -> List[CodeChunk]:

        return [
            chunk
            for chunk in self.chunks
            if chunk.symbol == symbol
        ]

    # ==================================================
    # GET BY TYPE
    # ==================================================

    def get_by_type(
        self,
        chunk_type: str
    ) -> List[CodeChunk]:

        return [
            chunk
            for chunk in self.chunks
            if chunk.chunk_type == chunk_type
        ]

    # ==================================================
    # SEARCH FILE + SYMBOL
    # ==================================================

    def find(
        self,
        file_path: Optional[str] = None,
        symbol: Optional[str] = None,
        chunk_type: Optional[str] = None,
    ) -> List[CodeChunk]:

        results = self.chunks

        if file_path:

            results = [
                chunk
                for chunk in results
                if chunk.file_path
                == file_path
            ]

        if symbol:

            results = [
                chunk
                for chunk in results
                if chunk.symbol
                == symbol
            ]

        if chunk_type:

            results = [
                chunk
                for chunk in results
                if chunk.chunk_type
                == chunk_type
            ]

        return results

    # ==================================================
    # STATISTICS
    # ==================================================

    def statistics(self) -> Dict:

        files = set()

        languages = {}

        chunk_types = {}

        symbols = 0

        for chunk in self.chunks:

            files.add(
                chunk.file_path
            )

            # ------------------------------
            # Languages
            # ------------------------------

            languages[
                chunk.language
            ] = (
                languages.get(
                    chunk.language,
                    0
                ) + 1
            )

            # ------------------------------
            # Chunk types
            # ------------------------------

            chunk_types[
                chunk.chunk_type
            ] = (
                chunk_types.get(
                    chunk.chunk_type,
                    0
                ) + 1
            )

            # ------------------------------
            # Symbols
            # ------------------------------

            if chunk.symbol:

                symbols += 1

        return {

            "total_chunks":
                len(self.chunks),

            "unique_files":
                len(files),

            "chunks_with_symbols":
                symbols,

            "languages":
                languages,

            "chunk_types":
                chunk_types,
        }

    # ==================================================
    # PRINT SUMMARY
    # ==================================================

    def print_summary(self):

        stats = (
            self.statistics()
        )

        print(
            "\n"
            "========================================"
        )

        print(
            "FINAL CHUNK STORE"
        )

        print(
            "========================================"
        )

        print(
            f"Total chunks: "
            f"{stats['total_chunks']}"
        )

        print(
            f"Unique files: "
            f"{stats['unique_files']}"
        )

        print(
            f"Chunks with symbols: "
            f"{stats['chunks_with_symbols']}"
        )

        print(
            "\nLANGUAGES"
        )

        for language, count in (
            stats["languages"].items()
        ):

            print(
                f"  {language}: {count}"
            )

        print(
            "\nCHUNK TYPES"
        )

        for chunk_type, count in (
            stats["chunk_types"].items()
        ):

            print(
                f"  {chunk_type}: {count}"
            )

    # ==================================================
    # SHOW CHUNK
    # ==================================================

    def show_chunk(
        self,
        chunk_id: str
    ):

        chunk = self.get_by_id(
            chunk_id
        )

        if not chunk:

            print(
                "\nChunk not found."
            )

            return

        print(
            "\n========================================"
        )

        print(
            "CHUNK"
        )

        print(
            "========================================"
        )

        print(
            f"ID: "
            f"{chunk.chunk_id}"
        )

        print(
            f"File: "
            f"{chunk.file_path}"
        )

        print(
            f"Type: "
            f"{chunk.chunk_type}"
        )

        print(
            f"Symbol: "
            f"{chunk.symbol}"
        )

        print(
            f"Lines: "
            f"{chunk.line_start}-"
            f"{chunk.line_end}"
        )

        print(
            f"Language: "
            f"{chunk.language}"
        )

        print(
            "\nContent:\n"
        )

        print(
            chunk.content
        )

        # ------------------------------------------
        # Relationships
        # ------------------------------------------

        print(
            "\nImports:"
        )

        for item in chunk.imports:

            print(
                f"  {item}"
            )

        print(
            "\nCalls:"
        )

        for item in chunk.calls:

            print(
                f"  {item}"
            )

        print(
            "\nDependencies:"
        )

        for item in chunk.dependencies:

            print(
                f"  {item}"
            )

        print(
            "\nRelated files:"
        )

        for item in chunk.related_files:

            print(
                f"  {item}"
            )

        # ------------------------------------------
        # GitHub reference
        # ------------------------------------------

        print(
            "\nGitHub reference:"
        )

        print(
            chunk.metadata.get(
                "github_reference"
            )
        )


# ======================================================
# TEST
# ======================================================

if __name__ == "__main__":

    storage_path = input(
        "Enter final chunks path: "
    ).strip()

    store = ChunkStore(
        storage_path
    )

    print(
        "\nLoading final chunks..."
    )

    store.load()

    store.print_summary()

    # ----------------------------------------------
    # Test file search
    # ----------------------------------------------

    print(
        "\n"
        "========================================"
    )

    print(
        "FILE SEARCH TEST"
    )

    print(
        "========================================"
    )

    file_path = input(
        "Enter file path to search "
        "(or press Enter to skip): "
    ).strip()

    if file_path:

        results = store.get_by_file(
            file_path
        )

        print(
            f"\nFound {len(results)} chunks."
        )

        for chunk in results:

            print(
                f"  {chunk.chunk_type:18}"
                f" {chunk.symbol}"
                f" "
                f"{chunk.line_start}-"
                f"{chunk.line_end}"
            )

    # ----------------------------------------------
    # Test symbol search
    # ----------------------------------------------

    print(
        "\n"
        "========================================"
    )

    print(
        "SYMBOL SEARCH TEST"
    )

    print(
        "========================================"
    )

    symbol = input(
        "Enter symbol to search "
        "(or press Enter to skip): "
    ).strip()

    if symbol:

        results = store.get_by_symbol(
            symbol
        )

        print(
            f"\nFound {len(results)} chunks."
        )

        for chunk in results:

            print(
                f"  {chunk.file_path}"
                f" "
                f"{chunk.line_start}-"
                f"{chunk.line_end}"
            )