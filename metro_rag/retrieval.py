"""Embed the knowledge base into a FAISS index and retrieve the most similar chunks for a query."""
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from .config import EMBEDDING_MODEL_NAME, TOP_K


def load_embedding_model(model_name=EMBEDDING_MODEL_NAME):
    return SentenceTransformer(model_name)


def build_index(embedding_model, knowledge_base, show_progress_bar=True):
    """
    E5 models are trained with `passage: ` / `query: ` prefixes. Vectors are normalized
    so that inner product equals cosine similarity.
    """
    passages = ["passage: " + text for text in knowledge_base["text_chunk"]]
    embeddings = embedding_model.encode(passages, normalize_embeddings=True, show_progress_bar=show_progress_bar)

    index = faiss.IndexFlatIP(embeddings.shape[1])  # inner product on normalized vectors = cosine similarity
    index.add(np.asarray(embeddings, dtype="float32"))
    return index


def retrieve_relevant_chunks(query, embedding_model, index, knowledge_base, k=TOP_K):
    """
    Takes a natural language query, embeds it, and retrieves the top-k
    most relevant chunks from the vector database.

    Parameters:
    - query (str): The user's search question.
    - embedding_model: The SentenceTransformer embedding model.
    - index: The FAISS vector database index.
    - knowledge_base (pd.DataFrame): The chunks, with 'chunk_id' and 'text_chunk' columns.
    - k (int): The number of results to return (default is TOP_K).

    Returns:
    - list of dicts: Contains the rank, chunk id, cosine similarity score, and text of the retrieved chunks.
    """
    # 1. Convert the text query into a normalized vector
    query_vector = embedding_model.encode(["query: " + query], normalize_embeddings=True)

    # 2. Search the FAISS index for the top-k most similar vectors
    scores, indices = index.search(np.asarray(query_vector, dtype="float32"), k)

    # 3. Compile the results
    return [
        {
            "rank": rank + 1,
            "chunk_id": knowledge_base["chunk_id"].iloc[idx],
            "score": round(float(score), 4),  # Higher score means higher relevance
            "text": knowledge_base["text_chunk"].iloc[idx],
        }
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0]))
    ]
