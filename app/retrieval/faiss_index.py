from pathlib import Path

import faiss
import numpy as np


class FAISSIndex:
    """
    FAISS vector index for repository embeddings.

    Uses:
        IndexFlatIP

    Because our embeddings are L2-normalized,
    inner product is equivalent to cosine similarity.
    """

    def __init__(
        self,
        dimension: int,
    ):

        self.dimension = dimension

        # ----------------------------------------------
        # Exact inner-product index
        # ----------------------------------------------

        self.index = faiss.IndexFlatIP(
            dimension
        )

    # ==================================================
    # ADD VECTORS
    # ==================================================

    def add(
        self,
        vectors: np.ndarray
    ):

        vectors = np.asarray(
            vectors,
            dtype=np.float32
        )

        if vectors.ndim != 2:

            raise ValueError(
                "Vectors must be 2D. "
                f"Got shape: {vectors.shape}"
            )

        if vectors.shape[1] != self.dimension:

            raise ValueError(
                "Vector dimension mismatch. "
                f"Expected {self.dimension}, "
                f"got {vectors.shape[1]}"
            )

        # ----------------------------------------------
        # Ensure normalized vectors
        # ----------------------------------------------

        faiss.normalize_L2(
            vectors
        )

        self.index.add(
            vectors
        )

    # ==================================================
    # SEARCH
    # ==================================================

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
    ):

        query_vector = np.asarray(
            query_vector,
            dtype=np.float32
        )

        if query_vector.ndim == 1:

            query_vector = (
                query_vector.reshape(
                    1,
                    -1
                )
            )

        if query_vector.shape[1] != self.dimension:

            raise ValueError(
                "Query dimension mismatch. "
                f"Expected {self.dimension}, "
                f"got {query_vector.shape[1]}"
            )

        faiss.normalize_L2(
            query_vector
        )

        distances, indices = (
            self.index.search(
                query_vector,
                top_k
            )
        )

        return distances, indices

    # ==================================================
    # SAVE
    # ==================================================

    def save(
        self,
        path: str | Path
    ):

        file_path = Path(path)

        file_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        faiss.write_index(
            self.index,
            str(file_path)
        )

        print(
            f"FAISS index saved:"
        )

        print(
            file_path
        )

    # ==================================================
    # LOAD
    # ==================================================

    @classmethod
    def load(
        cls,
        path: str | Path
    ):

        file_path = Path(path)

        if not file_path.exists():

            raise FileNotFoundError(
                f"FAISS index not found: "
                f"{file_path}"
            )

        index = faiss.read_index(
            str(file_path)
        )

        instance = cls(
            index.d
        )

        instance.index = index

        return instance

    # ==================================================
    # SIZE
    # ==================================================

    def size(self):

        return self.index.ntotal


# ======================================================
# TEST
# ======================================================

if __name__ == "__main__":

    print(
        "\n"
        "========================================"
    )

    print(
        "FAISS INDEX TEST"
    )

    print(
        "========================================"
    )

    embeddings_path = input(
        "Enter embeddings path: "
    ).strip()

    index_path = input(
        "Enter output FAISS index path: "
    ).strip()

    # ----------------------------------------------
    # Load embeddings
    # ----------------------------------------------

    vectors = np.load(
        embeddings_path
    )

    print(
        f"\nLoaded embeddings: "
        f"{vectors.shape}"
    )

    # ----------------------------------------------
    # Dimension
    # ----------------------------------------------

    dimension = vectors.shape[1]

    print(
        f"Vector dimension: "
        f"{dimension}"
    )

    # ----------------------------------------------
    # Build index
    # ----------------------------------------------

    faiss_index = FAISSIndex(
        dimension
    )

    faiss_index.add(
        vectors
    )

    print(
        f"Vectors in index: "
        f"{faiss_index.size()}"
    )

    # ----------------------------------------------
    # Save
    # ----------------------------------------------

    faiss_index.save(
        index_path
    )

    # ----------------------------------------------
    # Test search using first vector
    # ----------------------------------------------

    query_vector = vectors[0]

    distances, indices = (
        faiss_index.search(
            query_vector,
            top_k=5
        )
    )

    print(
        "\nTEST SEARCH"
    )

    print(
        "========================================"
    )

    for rank, (
        distance,
        index
    ) in enumerate(
        zip(
            distances[0],
            indices[0]
        ),
        start=1
    ):

        print(
            f"{rank}. "
            f"Vector index: {index} "
            f"| Similarity: "
            f"{distance:.4f}"
        )

    print(
        "\nFAISS test complete."
    )