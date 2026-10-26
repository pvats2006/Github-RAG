"""
Large retrieval benchmark for the Code Repository RAG.

Run:
    python -m app.retrieval.large_retrieval_evaluation

The benchmark evaluates retrieval only (no LLM calls).
It reports:
- File Recall@1/@3/@5
- Symbol Recall@1/@3/@5
- MRR for files and symbols
- Per-category metrics
- Failure cases
- JSON + CSV results
"""

import csv
import json
from pathlib import Path
from statistics import mean

from app.retrieval.semantic_retriever import SemanticRetriever


QUERIES = [
    # ---------------------------------------------------------
    # Training
    # ---------------------------------------------------------
    ("training", "Where is model training implemented?",
     "src/train.py", "train"),
    ("training", "Which function trains the classifier?",
     "src/train.py", "train"),
    ("training", "Where is the training pipeline defined?",
     "src/train.py", "train"),
    ("training", "Which function handles model training?",
     "src/train.py", "train"),
    ("training", "Where does model fitting happen?",
     "src/train.py", "train"),
    ("training", "What function runs the training stage?",
     "src/train.py", "train"),

    # ---------------------------------------------------------
    # Prediction
    # ---------------------------------------------------------
    ("prediction", "Where is prediction implemented?",
     "src/predict.py", "predict"),
    ("prediction", "Which function performs predictions?",
     "src/predict.py", "predict"),
    ("prediction", "Where is the prediction logic defined?",
     "src/predict.py", "predict"),
    ("prediction", "Which function calls the trained model for inference?",
     "src/predict.py", "predict"),
    ("prediction", "Where does inference happen?",
     "src/predict.py", "predict"),

    # ---------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------
    ("evaluation", "Where is model evaluation implemented?",
     "src/evaluate.py", "evaluate"),
    ("evaluation", "Which function evaluates the model?",
     "src/evaluate.py", "evaluate"),
    ("evaluation", "How is model performance measured?",
     "src/evaluate.py", "evaluate"),
    ("evaluation", "Where are classification metrics computed?",
     "src/evaluate.py", "evaluate"),

    # ---------------------------------------------------------
    # Feature extraction
    # ---------------------------------------------------------
    ("features", "Where is feature extraction implemented?",
     "src/features.py", "extract_features"),
    ("features", "Which function extracts image features?",
     "src/features.py", "extract_features"),
    ("features", "Where are HOG features extracted?",
     "src/features.py", "extract_hog"),
    ("features", "Which function computes HOG features?",
     "src/features.py", "extract_hog"),
    ("features", "Where is LBP feature extraction implemented?",
     "src/features.py", "extract_lbp"),
    ("features", "Which function normalizes feature vectors?",
     "src/features.py", "normalize_feature"),

    # ---------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------
    ("dataset", "Where is the dataset built?",
     "src/dataset_builder.py", "build_dataset"),
    ("dataset", "Which function builds the dataset?",
     "src/dataset_builder.py", "build_dataset"),
    ("dataset", "Where does the project read its training images?",
     "src/dataset_builder.py", "build_dataset"),
    ("dataset", "Which function creates the training dataset?",
     "src/dataset_builder.py", "build_dataset"),

    # ---------------------------------------------------------
    # Preprocessing
    # ---------------------------------------------------------
    ("preprocessing", "Where is image loading implemented?",
     "src/preprocess.py", "load_image"),
    ("preprocessing", "Which function loads an image?",
     "src/preprocess.py", "load_image"),
    ("preprocessing", "Where is image resizing implemented?",
     "src/preprocess.py", "resize_image"),
    ("preprocessing", "Which function resizes images?",
     "src/preprocess.py", "resize_image"),
    ("preprocessing", "Where is grayscale conversion implemented?",
     "src/preprocess.py", "convert_to_gray"),

    # ---------------------------------------------------------
    # API / web
    # ---------------------------------------------------------
    ("api", "Where is the prediction API endpoint defined?",
     "app/main.py", "predict"),
    ("api", "Which function handles prediction API requests?",
     "app/main.py", "predict"),
    ("api", "Where is the home endpoint defined?",
     "app/main.py", "home"),
    ("api", "Where is the image prediction route implemented?",
     "app/predictor.py", "predict_image"),
    ("api", "Which function processes the image for prediction?",
     "app/predictor.py", "predict_image"),

    # ---------------------------------------------------------
    # Utilities
    # ---------------------------------------------------------
    ("utility", "Where is directory creation implemented?",
     "src/utils.py", "create_directory"),
    ("utility", "Which function creates a directory?",
     "src/utils.py", "create_directory"),
    ("utility", "Where is the image listing function?",
     "src/utils.py", "list_images"),
    ("utility", "Which function lists images in a directory?",
     "src/utils.py", "list_images"),

    # ---------------------------------------------------------
    # Class lookup
    # ---------------------------------------------------------
    ("class_lookup", "Where is PredictionResponse defined?",
     "app/schemas.py", "PredictionResponse"),
    ("class_lookup", "Which class defines the API response schema?",
     "app/schemas.py", "PredictionResponse"),
]


def reciprocal_rank(items, expected, field):
    for rank, item in enumerate(items, 1):
        if item.get(field) == expected:
            return 1.0 / rank
    return 0.0


def hit_at(items, expected, field, k):
    return any(item.get(field) == expected for item in items[:k])


def main():
    print("=" * 60)
    print("LARGE RETRIEVAL EVALUATION")
    print("=" * 60)
    print(f"Benchmark queries: {len(QUERIES)}")
    print()

    # ------------------------------------------------------------------
    # Allow loading paths from a repository manifest (no manual input)
    # ------------------------------------------------------------------
    import sys
    import json
    from pathlib import Path

    index_path = metadata_path = chunks_path = ""

    # Check for --manifest flag: python -m app.retrieval.large_retrieval_evaluation --manifest <path>
    args = sys.argv[1:]
    if "--manifest" in args:
        manifest_idx = args.index("--manifest") + 1
        if manifest_idx < len(args):
            manifest_file = Path(args[manifest_idx])
            if manifest_file.exists():
                manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
                index_path    = manifest["index_path"]
                metadata_path = manifest["metadata_path"]
                chunks_path   = manifest["chunks_path"]
                print(f"Loaded paths from manifest: {manifest_file}")
                print(f"Repository: {manifest.get('repo_name', 'unknown')}")
                print()
            else:
                print(f"Manifest not found: {manifest_file}")

    # Fall back to manual path prompts if not loaded from manifest
    if not index_path:
        index_path = input(
            "Enter FAISS index path [data/embeddings/repository.index]: "
        ).strip() or "data/embeddings/repository.index"

    if not metadata_path:
        metadata_path = input(
            "Enter embedding metadata path [data/embeddings/embedding_metadata.json]: "
        ).strip() or "data/embeddings/embedding_metadata.json"

    if not chunks_path:
        chunks_path = input(
            "Enter final chunks path [data/chunks/final_chunks.json]: "
        ).strip() or "data/chunks/final_chunks.json"

    print("\nLoading retriever...")
    retriever = SemanticRetriever(
        index_path=index_path,
        metadata_path=metadata_path,
        chunks_path=chunks_path,
        top_k=5,
    )

    rows = []
    file_hits = {1: [], 3: [], 5: []}
    symbol_hits = {1: [], 3: [], 5: []}
    file_rrs = []
    symbol_rrs = []

    for number, (category, query, expected_file, expected_symbol) in enumerate(
        QUERIES, 1
    ):
        results = retriever.search(query, top_k=5)

        file_rank = None
        symbol_rank = None

        for rank, result in enumerate(results, 1):
            if file_rank is None and result.get("file_path") == expected_file:
                file_rank = rank
            if symbol_rank is None and result.get("symbol") == expected_symbol:
                symbol_rank = rank

        for k in (1, 3, 5):
            file_hits[k].append(file_rank is not None and file_rank <= k)
            symbol_hits[k].append(symbol_rank is not None and symbol_rank <= k)

        file_rrs.append(reciprocal_rank(results, expected_file, "file_path"))
        symbol_rrs.append(reciprocal_rank(results, expected_symbol, "symbol"))

        rows.append({
            "id": number,
            "category": category,
            "query": query,
            "expected_file": expected_file,
            "expected_symbol": expected_symbol,
            "file_rank": file_rank,
            "symbol_rank": symbol_rank,
            "file_top1": file_rank == 1,
            "file_top3": file_rank is not None and file_rank <= 3,
            "file_top5": file_rank is not None and file_rank <= 5,
            "symbol_top1": symbol_rank == 1,
            "symbol_top3": symbol_rank is not None and symbol_rank <= 3,
            "symbol_top5": symbol_rank is not None and symbol_rank <= 5,
            "top1_result": (
                f"{results[0].get('file_path')} | {results[0].get('symbol')}"
                if results else ""
            ),
        })

        status = "PASS" if symbol_rank == 1 and file_rank == 1 else "CHECK"
        print(
            f"[{number:02d}/{len(QUERIES)}] {status} | "
            f"{category:18s} | "
            f"F={file_rank or '-'} S={symbol_rank or '-'} | {query}"
        )

    print("\n" + "=" * 60)
    print("OVERALL METRICS")
    print("=" * 60)

    for k in (1, 3, 5):
        file_score = mean(file_hits[k]) * 100
        symbol_score = mean(symbol_hits[k]) * 100
        print(f"File Recall@{k}:    {sum(file_hits[k])}/{len(rows)} ({file_score:.2f}%)")
        print(f"Symbol Recall@{k}: {sum(symbol_hits[k])}/{len(rows)} ({symbol_score:.2f}%)")

    print(f"File MRR:           {mean(file_rrs):.4f}")
    print(f"Symbol MRR:         {mean(symbol_rrs):.4f}")

    print("\n" + "=" * 60)
    print("CATEGORY METRICS")
    print("=" * 60)

    categories = sorted(set(row["category"] for row in rows))
    for category in categories:
        group = [r for r in rows if r["category"] == category]
        symbol_ranks = [r["symbol_rank"] for r in group if r["symbol_rank"]]
        file_ranks = [r["file_rank"] for r in group if r["file_rank"]]

        symbol_top1 = sum(r["symbol_rank"] == 1 for r in group) / len(group) * 100
        file_top1 = sum(r["file_rank"] == 1 for r in group) / len(group) * 100

        print(
            f"{category:18s} | "
            f"File@1 {file_top1:6.2f}% | "
            f"Symbol@1 {symbol_top1:6.2f}% | "
            f"Queries {len(group)}"
        )

    failures = [
        r for r in rows
        if not r["file_top1"] or not r["symbol_top1"]
    ]

    print("\n" + "=" * 60)
    print(f"FAILURE ANALYSIS: {len(failures)} cases")
    print("=" * 60)

    if not failures:
        print("No Top-1 failures. Excellent.")
    else:
        for row in failures:
            print(f"\n[{row['id']}] {row['query']}")
            print(f"Expected file:   {row['expected_file']}")
            print(f"Actual file rank: {row['file_rank']}")
            print(f"Expected symbol: {row['expected_symbol']}")
            print(f"Actual symbol rank: {row['symbol_rank']}")
            print(f"Top-1 returned:  {row['top1_result']}")

    output_dir = Path("data/evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "large_retrieval_results.json"
    csv_path = output_dir / "large_retrieval_results.csv"

    summary = {
        "query_count": len(rows),
        "file_recall_at_1": mean(file_hits[1]),
        "file_recall_at_3": mean(file_hits[3]),
        "file_recall_at_5": mean(file_hits[5]),
        "symbol_recall_at_1": mean(symbol_hits[1]),
        "symbol_recall_at_3": mean(symbol_hits[3]),
        "symbol_recall_at_5": mean(symbol_hits[5]),
        "file_mrr": mean(file_rrs),
        "symbol_mrr": mean(symbol_rrs),
        "failures": len(failures),
        "rows": rows,
    }

    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print("\n" + "=" * 60)
    print("RESULT FILES")
    print("=" * 60)
    print(f"JSON: {json_path}")
    print(f"CSV:  {csv_path}")


if __name__ == "__main__":
    main()
