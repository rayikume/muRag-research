from loguru import logger
from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, model_name: str, expected_dim: int) -> None:
        logger.info("Loading embedding model: {}", model_name)
        self.model = SentenceTransformer(model_name)
        actual_dim = self.model.get_sentence_embedding_dimension()
        if actual_dim != expected_dim:
            raise ValueError(
                f"Embedding dim mismatch: model returns {actual_dim}, "
                f"config expects {expected_dim}. Update settings.embedding_dim "
                f"or pick a model with matching output dim."
            )
        self.dim = actual_dim
        logger.debug("Embedder ready (model={}, dim={})", model_name, self.dim)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]
