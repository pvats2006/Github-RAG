from typing import List

import torch
from sentence_transformers import SentenceTransformer

from app.chunking.chunk_schema import CodeChunk


class EmbeddingModel:
    """
    Local embedding model for repository RAG.

    Converts text/code chunks into dense vectors.
    """

    DEFAULT_MODEL = "all-MiniLM-L6-v2"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str | None = None,
    ):

        self.model_name = model_name

        # ----------------------------------------------
        # Automatically select best available device
        # ----------------------------------------------

        if device:

            self.device = device

        elif torch.backends.mps.is_available():

            self.device = "mps"

        elif torch.cuda.is_available():

            self.device = "cuda"

        else:

            self.device = "cpu"

        print(
            f"Loading embedding model: "
            f"{self.model_name}"
        )

        print(
            f"Device: {self.device}"
        )

        # ----------------------------------------------
        # Load model
        # ----------------------------------------------

        self.model = SentenceTransformer(
            self.model_name,
            device=self.device,
        )

        # ----------------------------------------------
        # Get vector dimension
        # ----------------------------------------------

        self.dimension = (
            self.model.get_embedding_dimension()
        )

        print(
            f"Embedding dimension: "
            f"{self.dimension}"
        )

    # ==================================================
    # ENCODE ONE TEXT
    # ==================================================

    def encode_text(
        self,
        text: str
    ):

        if not text or not text.strip():

            raise ValueError(
                "Cannot embed empty text"
            )

        vector = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return vector

    # ==================================================
    # ENCODE MANY TEXTS
    # ==================================================

    def encode_texts(
        self,
        texts: List[str],
        batch_size: int = 32,
    ):

        if not texts:

            return []

        vectors = self.model.encode(

            texts,

            batch_size=batch_size,

            convert_to_numpy=True,

            normalize_embeddings=True,

            show_progress_bar=True,
        )

        return vectors

    # ==================================================
    # BUILD CHUNK TEXT
    # ==================================================

    @staticmethod
    def build_chunk_text(
        chunk: CodeChunk
    ):

        """
        Build the text that will be embedded.

        We don't embed only raw code.

        We include useful structural metadata so
        semantic search understands what the chunk is.
        """

        parts = []

        # ----------------------------------------------
        # File
        # ----------------------------------------------

        parts.append(
            f"File: {chunk.file_path}"
        )

        # ----------------------------------------------
        # Language
        # ----------------------------------------------

        parts.append(
            f"Language: {chunk.language}"
        )

        # ----------------------------------------------
        # Type
        # ----------------------------------------------

        parts.append(
            f"Type: {chunk.chunk_type}"
        )

        # ----------------------------------------------
        # Symbol
        # ----------------------------------------------

        if chunk.symbol:

            parts.append(
                f"Symbol: {chunk.symbol}"
            )

        # ----------------------------------------------
        # Parent
        # ----------------------------------------------

        if chunk.parent_symbol:

            parts.append(
                f"Parent: "
                f"{chunk.parent_symbol}"
            )

        # ----------------------------------------------
        # Imports
        # ----------------------------------------------

        if chunk.imports:

            parts.append(
                "Imports: "
                + ", ".join(
                    chunk.imports
                )
            )

        # ----------------------------------------------
        # Calls
        # ----------------------------------------------

        if chunk.calls:

            parts.append(
                "Calls: "
                + ", ".join(
                    chunk.calls
                )
            )

        # ----------------------------------------------
        # Dependencies
        # ----------------------------------------------

        if chunk.dependencies:

            parts.append(
                "Dependencies: "
                + ", ".join(
                    chunk.dependencies
                )
            )

        # ----------------------------------------------
        # Related files
        # ----------------------------------------------

        if chunk.related_files:

            parts.append(
                "Related files: "
                + ", ".join(
                    chunk.related_files
                )
            )

        # ----------------------------------------------
        # Actual content
        # ----------------------------------------------

        parts.append(
            "Content:\n"
            + chunk.content
        )

        return "\n".join(
            parts
        )

    # ==================================================
    # ENCODE CHUNKS
    # ==================================================

    def encode_chunks(
        self,
        chunks: List[CodeChunk],
        batch_size: int = 32,
    ):

        if not chunks:

            return []

        texts = [
            self.build_chunk_text(
                chunk
            )
            for chunk in chunks
        ]

        vectors = self.encode_texts(
            texts,
            batch_size=batch_size,
        )

        return vectors


# ======================================================
# TEST
# ======================================================

if __name__ == "__main__":

    print(
        "\n"
        "========================================"
    )

    print(
        "EMBEDDING MODEL TEST"
    )

    print(
        "========================================"
    )

    model = EmbeddingModel()

    text = (
        "How does the machine learning "
        "project select the best model?"
    )

    vector = model.encode_text(
        text
    )

    print(
        "\nInput text:"
    )

    print(
        text
    )

    print(
        "\nVector shape:"
    )

    print(
        vector.shape
    )

    print(
        "\nFirst 10 values:"
    )

    print(
        vector[:10]
    )