from app.retrieval.semantic_retriever import SemanticRetriever


# ======================================================
# TEST DATASET
# ======================================================

TEST_QUERIES = [

    {
        "query": "How does the project select the best machine learning model?",
        "expected_file": "src/components/model_trainer.py",
        "expected_symbol": "ModelTrainer.initiate_model_trainer",
    },

    {
        "query": "Where is model training implemented?",
        "expected_file": "src/components/model_trainer.py",
        "expected_symbol": "ModelTrainer.initiate_model_trainer",
    },

    {
        "query": "Where is prediction implemented?",
        "expected_file": "src/pipeline/predict_pipeline.py",
        "expected_symbol": "PredictPipeline.predict",
    },

    {
        "query": "Where is data ingestion implemented?",
        "expected_file": "src/components/data_ingestion.py",
        "expected_symbol": "DataIngestion.initiate_data_ingestion",
    },

    {
        "query": "Where is data transformation implemented?",
        "expected_file": "src/components/data_transformation.py",
        "expected_symbol": "DataTransformation.initiate_data_transformation",
    },

    {
        "query": "How are models evaluated?",
        "expected_file": "src/utils.py",
        "expected_symbol": "evaluate_models",
    },

    {
        "query": "Where is the Flask prediction route defined?",
        "expected_file": "application.py",
        "expected_symbol": "predict_datapoint",
    },

    {
        "query": "How does the application create prediction data?",
        "expected_file": "src/pipeline/predict_pipeline.py",
        "expected_symbol": "CustomData.get_data_as_data_frame",
    },

    {
        "query": "Where are models saved?",
        "expected_file": "src/components/model_trainer.py",
        "expected_symbol": "ModelTrainer.initiate_model_trainer",
    },

    {
        "query": "Where is the custom exception defined?",
        "expected_file": "src/exception.py",
        "expected_symbol": "CustomException",
    },
]


# ======================================================
# MATCH HELPERS
# ======================================================

def file_match(
    result,
    expected_file,
):

    return (
        result.get("file_path")
        == expected_file
    )


def symbol_match(
    result,
    expected_symbol,
):

    return (
        result.get("symbol")
        == expected_symbol
    )


# ======================================================
# EVALUATE ONE QUERY
# ======================================================

def evaluate_query(
    retriever,
    test_case,
):

    query = test_case["query"]

    expected_file = test_case[
        "expected_file"
    ]

    expected_symbol = test_case[
        "expected_symbol"
    ]

    results = retriever.search(
        query,
        top_k=5,
    )

    top1 = results[:1]
    top3 = results[:3]
    top5 = results[:5]

    # ----------------------------------------------
    # File retrieval
    # ----------------------------------------------

    top1_file = any(
        file_match(
            result,
            expected_file,
        )
        for result in top1
    )

    top3_file = any(
        file_match(
            result,
            expected_file,
        )
        for result in top3
    )

    top5_file = any(
        file_match(
            result,
            expected_file,
        )
        for result in top5
    )

    # ----------------------------------------------
    # Symbol retrieval
    # ----------------------------------------------

    top1_symbol = any(
        symbol_match(
            result,
            expected_symbol,
        )
        for result in top1
    )

    top3_symbol = any(
        symbol_match(
            result,
            expected_symbol,
        )
        for result in top3
    )

    top5_symbol = any(
        symbol_match(
            result,
            expected_symbol,
        )
        for result in top5
    )

    return {
        "query": query,

        "expected_file": expected_file,

        "expected_symbol": expected_symbol,

        "top1_file": top1_file,
        "top3_file": top3_file,
        "top5_file": top5_file,

        "top1_symbol": top1_symbol,
        "top3_symbol": top3_symbol,
        "top5_symbol": top5_symbol,

        "results": results,
    }


# ======================================================
# MAIN EVALUATION
# ======================================================

def run_evaluation(
    retriever,
):

    total = len(
        TEST_QUERIES
    )

    file_top1 = 0
    file_top3 = 0
    file_top5 = 0

    symbol_top1 = 0
    symbol_top3 = 0
    symbol_top5 = 0

    print(
        "\n"
        "========================================"
    )

    print(
        "RETRIEVAL EVALUATION"
    )

    print(
        "========================================"
    )

    print(
        f"\nTest queries: {total}"
    )

    for index, test_case in enumerate(
        TEST_QUERIES,
        start=1,
    ):

        evaluation = evaluate_query(
            retriever,
            test_case,
        )

        if evaluation["top1_file"]:
            file_top1 += 1

        if evaluation["top3_file"]:
            file_top3 += 1

        if evaluation["top5_file"]:
            file_top5 += 1

        if evaluation["top1_symbol"]:
            symbol_top1 += 1

        if evaluation["top3_symbol"]:
            symbol_top3 += 1

        if evaluation["top5_symbol"]:
            symbol_top5 += 1

        print(
            f"\n[{index}/{total}]"
        )

        print(
            f"Query: "
            f"{test_case['query']}"
        )

        print(
            f"Expected file: "
            f"{test_case['expected_file']}"
        )

        print(
            f"Expected symbol: "
            f"{test_case['expected_symbol']}"
        )

        print(
            "\nFile retrieval:"
        )

        print(
            f"  Top-1: "
            f"{'PASS' if evaluation['top1_file'] else 'FAIL'}"
        )

        print(
            f"  Top-3: "
            f"{'PASS' if evaluation['top3_file'] else 'FAIL'}"
        )

        print(
            f"  Top-5: "
            f"{'PASS' if evaluation['top5_file'] else 'FAIL'}"
        )

        print(
            "\nSymbol retrieval:"
        )

        print(
            f"  Top-1: "
            f"{'PASS' if evaluation['top1_symbol'] else 'FAIL'}"
        )

        print(
            f"  Top-3: "
            f"{'PASS' if evaluation['top3_symbol'] else 'FAIL'}"
        )

        print(
            f"  Top-5: "
            f"{'PASS' if evaluation['top5_symbol'] else 'FAIL'}"
        )

        print(
            "\nRetrieved:"
        )

        for rank, result in enumerate(
            evaluation["results"],
            start=1,
        ):

            print(
                f"  {rank}. "
                f"{result['file_path']} "
                f"| "
                f"{result['symbol']} "
                f"| "
                f"{result['rerank_score']:.4f}"
            )

    # ==================================================
    # FINAL METRICS
    # ==================================================

    def percentage(
        value,
    ):

        return (
            value / total
        ) * 100

    print(
        "\n"
        "========================================"
    )

    print(
        "FINAL RETRIEVAL METRICS"
    )

    print(
        "========================================"
    )

    print(
        "\nFILE RETRIEVAL"
    )

    print(
        f"Top-1: "
        f"{file_top1}/{total} "
        f"({percentage(file_top1):.2f}%)"
    )

    print(
        f"Top-3: "
        f"{file_top3}/{total} "
        f"({percentage(file_top3):.2f}%)"
    )

    print(
        f"Top-5: "
        f"{file_top5}/{total} "
        f"({percentage(file_top5):.2f}%)"
    )

    print(
        "\nSYMBOL RETRIEVAL"
    )

    print(
        f"Top-1: "
        f"{symbol_top1}/{total} "
        f"({percentage(symbol_top1):.2f}%)"
    )

    print(
        f"Top-3: "
        f"{symbol_top3}/{total} "
        f"({percentage(symbol_top3):.2f}%)"
    )

    print(
        f"Top-5: "
        f"{symbol_top5}/{total} "
        f"({percentage(symbol_top5):.2f}%)"
    )

    print(
        "\n"
        "========================================"
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
        "RETRIEVAL EVALUATION TEST"
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

    retriever = SemanticRetriever(
        index_path=index_path,
        metadata_path=metadata_path,
        chunks_path=chunks_path,
        top_k=5,
    )

    run_evaluation(
        retriever
    )