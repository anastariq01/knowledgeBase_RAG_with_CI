import json
import pickle
import numpy as np
import faiss


def load_json(filepath):
    """
    Load knowledge base JSON.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def save_pickle(obj, filepath):
    """
    Save Python object.
    """
    with open(filepath, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(filepath):
    """
    Load Python object.
    """
    with open(filepath, "rb") as f:
        return pickle.load(f)


def save_faiss_index(index, filepath):
    """
    Save FAISS index.
    """
    faiss.write_index(index, filepath)


def load_faiss_index(filepath):
    """
    Load FAISS index.
    """
    return faiss.read_index(filepath)


def normalize_embeddings(embeddings):
    """
    Normalize vectors for cosine similarity.
    """
    faiss.normalize_L2(embeddings)
    return embeddings


def search(index, query_embedding, k=3):
    """
    Search top-k nearest neighbors.
    """
    distances, indices = index.search(
        query_embedding,
        k
    )

    return distances, indices