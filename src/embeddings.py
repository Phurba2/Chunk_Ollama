import logging
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer
from config.settings import (
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_DEVICE,
    EMBEDDING_MODEL,
)


logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generate embeddings using Sentence Transformers."""

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(
                EmbeddingGenerator,
                cls
            ).__new__(cls)

            cls._instance._initialized = False

        return cls._instance

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL,
        device: str = EMBEDDING_DEVICE,
    ):
        if self._initialized:
            return

        logger.info(
            "Loading embedding model: %s",
            model_name,
        )

        self.model_name = model_name
        self.device = device

        self.model = SentenceTransformer(
            model_name,
            device=device,
        )

        self.embedding_dimension = (
            self.model.get_sentence_embedding_dimension()
        )

        logger.info(
            "Embedding dimension: %d",
            self.embedding_dimension,
        )

        self._initialized = True

    def generate_embeddings(
        self,
        texts: List[str],
        show_progress: bool = True,
    ) -> np.ndarray:
        """Generate embeddings for a list of texts."""

        if not texts:
            return np.empty(
                (0, self.embedding_dimension),
                dtype=np.float32,
            )

        cleaned_texts = [
            self._preprocess_text(text)
            for text in texts
        ]

        embeddings = self.model.encode(
            cleaned_texts,
            batch_size=EMBEDDING_BATCH_SIZE,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return embeddings.astype(
            np.float32
        )

    def _preprocess_text(
        self,
        text: str,
    ) -> str:
        """Clean text before embedding generation."""

        if not text:
            return ""

        # Normalize whitespace.
        text = " ".join(
            text.split()
        )

        return text.strip()

    def generate_query_embedding(
        self,
        query: str,
    ) -> np.ndarray:
        """Generate an embedding for a search query."""

        embeddings = self.generate_embeddings(
            [query],
            show_progress=False,
        )

        return embeddings[0]