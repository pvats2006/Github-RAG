import json
from pathlib import Path

import numpy as np

from app.chunking.chunk_schema import CodeChunk
from app.chunking.chunk_store import ChunkStore
from app.embeddings.embedding_model import EmbeddingModel


class ChunkEmbeddingPipeline:
    """
    Convert validated repository chunks into
    dense embedding vectors.

    Input:
        final_chunks.json

    Output:
        embeddings.npy
        embedding_metadata.json
    """

    def __init__(
        self,
        chunks_path: str,
        output_dir: str = "data/embeddings",
    ):

        self.chunks_path = Path(
            chunks_path
        )

        self.output_dir = Path(
            output_dir
        )

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        self.embedding_model = (
            EmbeddingModel()
        )

    # ==================================================
    # LOAD CHUNKS
    # ==================================================

    def load_chunks(self):

        print(
            "\nLoading validated chunks..."
        )

        store = ChunkStore(
            str(self.chunks_path)
        )

        chunks = store.load()

        print(
            f"Loaded chunks: {len(chunks)}"
        )

        return chunks

    # ==================================================
    # GENERATE EMBEDDINGS
    # ==================================================

    def generate_embeddings(
        self,
        chunks,
        batch_size: int = 32,
    ):

        print(
            "\nGenerating embeddings..."
        )

        vectors = (
            self.embedding_model.encode_chunks(
                chunks,
                batch_size=batch_size
            )
        )

        vectors = np.asarray(
            vectors,
            dtype=np.float32
        )

        print(
            f"Embedding matrix shape: "
            f"{vectors.shape}"
        )

        return vectors

    # ==================================================
    # SAVE VECTORS
    # ==================================================

    def save_embeddings(
        self,
        vectors
    ):

        output_path = (
            self.output_dir
            / "embeddings.npy"
        )

        np.save(
            output_path,
            vectors
        )

        print(
            f"\nEmbeddings saved:"
        )

        print(
            output_path
        )

        return output_path

    # ==================================================
    # SAVE METADATA
    # ==================================================

    def save_metadata(
        self,
        chunks
    ):

        """
        Save the mapping:

            vector index
                 ↓
            chunk_id

        FAISS will later return an integer index.
        This metadata lets us map that index back
        to the original CodeChunk.
        """

        metadata = []

        for index, chunk in enumerate(
            chunks
        ):

            metadata.append({

                "vector_index": index,

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

                "github_reference":
                    chunk.metadata.get(
                        "github_reference"
                    ),
            })

        output_path = (
            self.output_dir
            / "embedding_metadata.json"
        )

        with open(
            output_path,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                metadata,
                f,
                indent=2,
                ensure_ascii=False
            )

        print(
            "\nEmbedding metadata saved:"
        )

        print(
            output_path
        )

        return output_path

    # ==================================================
    # RUN PIPELINE
    # ==================================================

    def run(self):

        print(
            "\n"
            "========================================"
        )

        print(
            "CHUNK EMBEDDING PIPELINE"
        )

        print(
            "========================================"
        )

        # ----------------------------------------------
        # 1. Load chunks
        # ----------------------------------------------

        chunks = self.load_chunks()

        if not chunks:

            raise ValueError(
                "No chunks found."
            )

        # ----------------------------------------------
        # 2. Generate embeddings
        # ----------------------------------------------

        vectors = (
            self.generate_embeddings(
                chunks
            )
        )

        # ----------------------------------------------
        # 3. Validate shape
        # ----------------------------------------------

        expected_dimension = (
            self.embedding_model.dimension
        )

        if vectors.shape != (
            len(chunks),
            expected_dimension
        ):

            raise ValueError(
                "Embedding shape mismatch. "
                f"Expected "
                f"({len(chunks)}, "
                f"{expected_dimension}), "
                f"got {vectors.shape}"
            )

        # ----------------------------------------------
        # 4. Save vectors
        # ----------------------------------------------

        embeddings_path = (
            self.save_embeddings(
                vectors
            )
        )

        # ----------------------------------------------
        # 5. Save metadata
        # ----------------------------------------------

        metadata_path = (
            self.save_metadata(
                chunks
            )
        )

        # ----------------------------------------------
        # Final summary
        # ----------------------------------------------

        print(
            "\n"
            "========================================"
        )

        print(
            "EMBEDDING PIPELINE COMPLETE"
        )

        print(
            "========================================"
        )

        print(
            f"Chunks: "
            f"{len(chunks)}"
        )

        print(
            f"Dimensions: "
            f"{expected_dimension}"
        )

        print(
            f"Matrix: "
            f"{vectors.shape}"
        )

        print(
            f"\nVectors:"
        )

        print(
            embeddings_path
        )

        print(
            f"\nMetadata:"
        )

        print(
            metadata_path
        )


# ======================================================
# MAIN
# ======================================================

if __name__ == "__main__":

    chunks_path = input(
        "Enter final chunks path: "
    ).strip()

    pipeline = ChunkEmbeddingPipeline(
        chunks_path
    )

    pipeline.run()