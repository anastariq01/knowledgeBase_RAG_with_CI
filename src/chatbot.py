import numpy as np

from sentence_transformers import (
    SentenceTransformer
)

from utils import (
    load_pickle,
    load_faiss_index,
    normalize_embeddings,
    search
)

MODEL_NAME = "all-MiniLM-L6-v2"

INDEX_PATH = "D:/PROGRAMMING/AI/uni_ai_chatbot/embeddings/faiss.index"

METADATA_PATH = "D:/PROGRAMMING/AI/uni_ai_chatbot/embeddings/metadata.pkl"


def print_top_matches(
    metadata,
    distances,
    indices
):
    print("\nTop Matches:")

    for rank, idx in enumerate(
        indices[0],
        start=1
    ):

        item = metadata[idx]

        score = round(
            float(distances[0][rank - 1]),
            3
        )

        print(
            f"\n[{rank}] Similarity: {score}"
        )

        print(
            f"Q: {item['question']}"
        )

        print(
            f"A: {item['answer']}"
        )


def main():

    print(
        "\nLoading Father Profession Chatbot..."
    )

    model = SentenceTransformer(
        MODEL_NAME
    )

    index = load_faiss_index(
        INDEX_PATH
    )

    metadata = load_pickle(
        METADATA_PATH
    )

    print(
        "\nChatbot Ready!"
    )

    print(
        "Type 'exit' to quit.\n"
    )

    while True:

        query = input(
            "You: "
        ).strip()

        if query.lower() == "exit":
            print(
                "\nGoodbye!"
            )
            break

        query_embedding = model.encode(
            [query],
            convert_to_numpy=True
        )

        query_embedding = (
            query_embedding.astype(
                "float32"
            )
        )

        query_embedding = (
            normalize_embeddings(
                query_embedding
            )
        )

        distances, indices = search(
            index,
            query_embedding,
            k=3
        )

        best_score = float(
            distances[0][0]
        )

        best_answer = metadata[
            indices[0][0]
        ]

        print("\nBot:")
        print(
            best_answer["answer"]
        )

        print(
            f"\nConfidence: {best_score:.3f}"
        )

        show_matches = input(
            "\nShow top matches? (y/n): "
        ).lower()

        if show_matches == "y":
            print_top_matches(
                metadata,
                distances,
                indices
            )

        print("\n" + "-" * 60)


if __name__ == "__main__":
    main()