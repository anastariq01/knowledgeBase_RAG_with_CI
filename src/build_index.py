import numpy as np
import faiss

from sentence_transformers import SentenceTransformer

from utils import (    
    load_json,
    save_pickle,
    save_faiss_index,
    normalize_embeddings
)

MODEL_NAME = "all-MiniLM-L6-v2"

DATA_PATH = "D:/PROGRAMMING/AI/uni_ai_chatbot/data/knowledge.json"

INDEX_PATH = "D:/PROGRAMMING/AI/uni_ai_chatbot/embeddings/faiss.index"

METADATA_PATH = "D:/PROGRAMMING/AI/uni_ai_chatbot/embeddings/metadata.pkl"


def main():

    print("Loading knowledge base...")

    data = load_json(DATA_PATH)

    questions = [
        item["question"]
        for item in data
    ]

    print(
        f"Loaded {len(questions)} questions."
    )

    print("Loading embedding model...")

    model = SentenceTransformer(
        MODEL_NAME
    )

    print("Generating embeddings...")

    embeddings = model.encode(
        questions,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    embeddings = embeddings.astype(
        "float32"
    )

    embeddings = normalize_embeddings(
        embeddings
    )

    dimension = embeddings.shape[1]

    print(
        f"Embedding dimension: {dimension}"
    )

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(embeddings)

    save_faiss_index(
        index,
        INDEX_PATH
    )

    save_pickle(
        data,
        METADATA_PATH
    )

    print("\nIndex built successfully.")
    print(
        f"Questions indexed: {len(data)}"
    )


if __name__ == "__main__":
    main()