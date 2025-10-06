from loguru import logger
from sentence_transformers import SentenceTransformer


class Embedder:
    """Wrapper around sentence-transformers to provide a consistent interface for embedding texts."""

    ##TODO: Load from config
    def __init__(
        self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", normalize: bool = True
    ):
        logger.info("Loading embedding model: {name}", name=model_name)
        self.model = SentenceTransformer(model_name)
        self.normalize = normalize
        logger.info("Embedder ready; dim={d}", d=self.dim)

    def encode(self, texts):
        logger.debug("Encoding {n} texts", n=len(texts) if hasattr(texts, "__len__") else 1)
        return self.model.encode(texts, normalize_embeddings=self.normalize).tolist()

    @property
    def dim(self) -> int:
        return self.model.get_sentence_embedding_dimension()
