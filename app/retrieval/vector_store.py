import json
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np

from app.chunking.chunk_schema import CodeChunk
from app.chunking.chunk_store import ChunkStore
from app.retrieval.faiss_index import FAISSIndex


class VectorStore:
    """
    Connects:

        FAISS
          +
        Embedding Metadata
          +
        Final Chunks

    so that a FAISS vector index can be converted
    back into real repository chunks.
    """

    def __init__(
        self,
        index_path: str,
        metadata_path: str,
        chunks_path: str,
    ):

        self.index_path = Path(
            index_path
        )

        self.metadata_path = Path(
            metadata_path
        )

        self.chunks_path = Path(
            chunks_path
        )

        self.index: Optional[FAISSIndex] = None

        self.metadata: List[Dict] = []

        self.chunk_store: Optional[ChunkStore] = None

        self.vector_to_chunk_id: Dict = {}

    # ==================================================
    # LOAD
    # ==================================================

    def load(self):

        print(
            "\nLoading vector store..."
        )

        # ----------------------------------------------
        # Load FAISS
        # ----------------------------------------------

        self.index = FAISSIndex.load(
            str(self.index_path)
        )

        print(
            f"FAISS vectors: "
            f"{self.index.size()}"
        )

        # ----------------------------------------------
        # Load embedding metadata
        # ----------------------------------------------

        if not self.metadata_path.exists():

            raise FileNotFoundError(
                f"Metadata not found: "
                f"{self.metadata_path}"
            )

        with open(
            self.metadata_path,
            "r",
            encoding="utf-8"
        ) as f:

            self.metadata = json.load(
                f
            )

        print(
            f"Metadata records: "
            f"{len(self.metadata)}"
        )

        # ----------------------------------------------
        # Load actual chunks
        # ----------------------------------------------

        self.chunk_store = ChunkStore(
            str(self.chunks_path)
        )

        self.chunk_store.load()

        print(
            f"Repository chunks: "
            f"{len(self.chunk_store.chunks)}"
        )

        # ----------------------------------------------
        # Build vector → chunk mapping
        # ----------------------------------------------

        self.vector_to_chunk_id = {}

        for record in self.metadata:

            vector_index = record[
                "vector_index"
            ]

            chunk_id = record[
                "chunk_id"
            ]

            self.vector_to_chunk_id[
                vector_index
            ] = chunk_id

        # ----------------------------------------------
        # Integrity checks
        # ----------------------------------------------

        if self.index.size() != len(
            self.metadata
        ):

            raise ValueError(
                "FAISS index and metadata "
                "size mismatch. "
                f"FAISS={self.index.size()} "
                f"metadata={len(self.metadata)}"
            )

        if self.index.size() != len(
            self.chunk_store.chunks
        ):

            raise ValueError(
                "FAISS index and chunk store "
                "size mismatch. "
                f"FAISS={self.index.size()} "
                f"chunks={len(self.chunk_store.chunks)}"
            )

        print(
            "Vector store loaded successfully."
        )

    # ==================================================
    # GET METADATA
    # ==================================================

    def get_metadata(
        self,
        vector_index: int
    ):

        for record in self.metadata:

            if record[
                "vector_index"
            ] == vector_index:

                return record

        return None

    # ==================================================
    # GET CHUNK
    # ==================================================

    def get_chunk(
        self,
        vector_index: int
    ):

        chunk_id = (
            self.vector_to_chunk_id.get(
                vector_index
            )
        )

        if not chunk_id:

            return None

        assert self.chunk_store is not None, (
            "VectorStore.load() must be called first."
        )
        return self.chunk_store.get_by_id(
            chunk_id
        )

    # ==================================================
    # FORMAT RESULT
    # ==================================================

    def format_result(
        self,
        vector_index: int,
        similarity: float
    ):

        chunk = self.get_chunk(
            vector_index
        )

        metadata = self.get_metadata(
            vector_index
        )

        if chunk is None:

            return None

        return {

            "vector_index":
                vector_index,

            "similarity":
                float(similarity),

            "chunk_id":
                chunk.chunk_id,

            "repo":
                chunk.repo,

            "file_path":
                chunk.file_path,

            "file_name":
                chunk.file_name,

            "language":
                chunk.language,

            "chunk_type":
                chunk.chunk_type,

            "symbol":
                chunk.symbol,

            "line_start":
                chunk.line_start,

            "line_end":
                chunk.line_end,

            "content":
                chunk.content,

            "imports":
                chunk.imports,

            "calls":
                chunk.calls,

            "dependencies":
                chunk.dependencies,

            "related_files":
                chunk.related_files,

            "parent_symbol":
                chunk.parent_symbol,

            "github_reference":
                (
                    chunk.metadata.get(
                        "github_reference"
                    )
                    or
                    (
                        metadata or {}
                    ).get(
                        "github_reference"
                    )
                ),
        }

    # ==================================================
    # SEARCH
    # ==================================================

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
    ):

        if self.index is None:

            raise RuntimeError(
                "VectorStore is not loaded."
            )

        # ----------------------------------------------
        # Don't request more results than exist
        # ----------------------------------------------

        top_k = min(
            top_k,
            self.index.size()
        )

        distances, indices = (
            self.index.search(
                query_vector,
                top_k
            )
        )

        results = []

        for similarity, vector_index in zip(
            distances[0],
            indices[0]
        ):

            # FAISS can return -1 when no result
            if vector_index < 0:

                continue

            result = (
                self.format_result(
                    int(vector_index),
                    float(similarity)
                )
            )

            if result:

                results.append(
                    result
                )

        return results

    # ==================================================
    # PRINT RESULTS
    # ==================================================

    @staticmethod
    def print_results(
        results: List[Dict]
    ):

        print(
            "\n"
            "========================================"
        )

        print(
            "VECTOR SEARCH RESULTS"
        )

        print(
            "========================================"
        )

        for rank, result in enumerate(
            results,
            start=1
        ):

            print(
                f"\n[{rank}] "
                f"Similarity: "
                f"{result['similarity']:.4f}"
            )

            print(
                f"File: "
                f"{result['file_path']}"
            )

            print(
                f"Type: "
                f"{result['chunk_type']}"
            )

            print(
                f"Symbol: "
                f"{result['symbol']}"
            )

            print(
                f"Lines: "
                f"{result['line_start']}-"
                f"{result['line_end']}"
            )

            print(
                f"Chunk ID: "
                f"{result['chunk_id']}"
            )

            print(
                f"GitHub: "
                f"{result['github_reference']}"
            )

            print(
                "Content preview:"
            )

            preview = (
                result["content"]
                .strip()
                .replace(
                    "\n",
                    " "
                )
            )

            if len(preview) > 250:

                preview = (
                    preview[:250]
                    + "..."
                )

            print(
                f"  {preview}"
            )


# ======================================================
# TEST
# ======================================================

if __name__ == "__main__":

    print(
        "\n"
        "========================================"
    )

    print(
        "VECTOR STORE TEST"
    )

    print(
        "========================================"
    )

    index_path = input(
        "Enter FAISS index path: "
    ).strip()

    metadata_path = input(
        "Enter embedding metadata path: "
    ).strip()

    chunks_path = input(
        "Enter final chunks path: "
    ).strip()

    # ----------------------------------------------
    # Load
    # ----------------------------------------------

    store = VectorStore(

        index_path=index_path,

        metadata_path=metadata_path,

        chunks_path=chunks_path,
    )

    store.load()

    # ----------------------------------------------
    # Test vector search
    # ----------------------------------------------

    test_index = 0

    assert store.index is not None, "store.load() must be called first."

    test_vector = np.array(
        store.index.index.reconstruct(test_index),
        dtype=np.float32,
    )

    results = store.search(
        test_vector,
        top_k=5
    )

    store.print_results(
        results
    )