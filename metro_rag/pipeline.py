"""Builds the full RAG system once and exposes retrieval and question answering."""
from .config import TOP_K
from .data_processing import prepare_stations
from .knowledge_base import build_knowledge_base
from .llm import generate_answer
from .retrieval import build_index, load_embedding_model, retrieve_relevant_chunks


class MetroRAG:
    def __init__(self, show_progress_bar=True):
        self.df = prepare_stations()
        self.knowledge_base = build_knowledge_base(self.df)
        self.embedding_model = load_embedding_model()
        self.index = build_index(self.embedding_model, self.knowledge_base, show_progress_bar=show_progress_bar)

    def retrieve(self, query, k=TOP_K):
        return retrieve_relevant_chunks(query, self.embedding_model, self.index, self.knowledge_base, k=k)

    def ask(self, user_question, retrieved=None, k=TOP_K):
        # Callers that already retrieved can pass the results in
        if retrieved is None:
            retrieved = self.retrieve(user_question, k=k)
        return generate_answer(user_question, retrieved)
