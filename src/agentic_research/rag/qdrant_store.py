import hashlib
from typing import Dict, List, Optional

from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams

from ..config.config import settings
from .embeddings import Embedder


class QdrantRAG:
    def __init__(self, collection: Optional[str] = None):
        self.collection = collection or settings.collection_name
        self.embedder = Embedder()
        logger.info("Initializing QdrantRAG collection={col}", col=self.collection)
        # Client: in-memory or server
        if settings.qdrant_url:
            self.client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
        else:
            # in-memory (embedded) if available; falls back to file-less mode
            self.client = QdrantClient(location=":memory:")
        self._ensure_collection()

    def _ensure_collection(self):
        try:
            self.client.get_collection(self.collection)
            logger.debug("Qdrant collection exists: {col}", col=self.collection)
        except Exception:
            logger.info(
                "Creating Qdrant collection: {col} dim={d}",
                col=self.collection,
                d=self.embedder.dim,
            )
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=self.embedder.dim, distance=Distance.COSINE),
            )

    @staticmethod
    def _ids(texts: List[str]) -> List[str]:
        return [hashlib.md5(t.encode()).hexdigest() for t in texts]

    def upsert(self, texts: List[str], metas: List[Dict]):
        if not texts:
            return
        vecs = self.embedder.encode(texts)
        points = []
        for v, t, m, pid in zip(vecs, texts, metas, self._ids(texts)):
            payload = {"text": t, **m}
            points.append(PointStruct(id=pid, vector=v, payload=payload))
        logger.info(
            "Upserting {n} points into collection {col}", n=len(points), col=self.collection
        )
        self.client.upsert(collection_name=self.collection, points=points)

    def query(self, q: str, k: int = 6):
        logger.info("RAG query: '{q}' k={k}", q=q, k=k)
        qv = self.embedder.encode([q])[0]
        res = self.client.search(
            collection_name=self.collection, query_vector=qv, limit=k, with_payload=True
        )
        out = []
        for r in res:
            p = r.payload or {}
            out.append(
                {
                    "text": p.get("text"),
                    "url": p.get("url"),
                    "title": p.get("title"),
                    "score": r.score,
                }
            )
        logger.debug("RAG returned {n} hits for query '{q}'", n=len(out), q=q)
        return out
