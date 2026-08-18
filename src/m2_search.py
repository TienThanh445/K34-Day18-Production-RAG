from __future__ import annotations

"""Module 2: Hybrid Search — BM25 (Vietnamese) + Dense + RRF."""

import os, sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME, EMBEDDING_MODEL,
                    EMBEDDING_DIM, BM25_TOP_K, DENSE_TOP_K, HYBRID_TOP_K)


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    method: str  # "bm25", "dense", "hybrid"


def segment_vietnamese(text: str) -> str:
    """Segment Vietnamese text into words."""
    try:
        from underthesea import word_tokenize
        segmented = word_tokenize(text, format="text")
        return segmented.replace("_", " ")
    except Exception:
        return text


class BM25Search:
    def __init__(self):
        self.corpus_tokens = []
        self.documents = []
        self.bm25 = None

    def index(self, chunks: list[dict]) -> None:
        """Build BM25 index from chunks."""
        from rank_bm25 import BM25Okapi

        self.documents = chunks
        self.corpus_tokens = []
        for chunk in chunks:
            text = chunk.get("text", "")
            tokens = segment_vietnamese(text).lower().split()
            self.corpus_tokens.append(tokens)

        if self.corpus_tokens:
            self.bm25 = BM25Okapi(self.corpus_tokens)
        else:
            self.bm25 = None

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[SearchResult]:
        """Search using BM25."""
        if self.bm25 is None or not self.documents:
            return []

        tokenized_query = segment_vietnamese(query).lower().split()
        if not tokenized_query:
            return []

        scores = self.bm25.get_scores(tokenized_query)
        sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        results = []
        for i in sorted_indices:
            score = float(scores[i])
            if score > 0:
                doc = self.documents[i]
                results.append(SearchResult(
                    text=doc.get("text", ""),
                    score=score,
                    metadata=doc.get("metadata", {}),
                    method="bm25"
                ))
            if len(results) >= top_k:
                break

        return results


class DenseSearch:
    def __init__(self):
        from qdrant_client import QdrantClient
        self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        self._encoder = None

    def _get_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(EMBEDDING_MODEL)
        return self._encoder

    def index(self, chunks: list[dict], collection: str = COLLECTION_NAME) -> None:
        """Index chunks into Qdrant."""
        from qdrant_client.models import Distance, VectorParams, PointStruct

        self.client.recreate_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE)
        )
        if not chunks:
            return

        texts = [c["text"] for c in chunks]
        vectors = self._get_encoder().encode(texts, show_progress_bar=False)

        points = []
        for i, (c, v) in enumerate(zip(chunks, vectors)):
            payload = dict(c.get("metadata", {}))
            payload["text"] = c["text"]
            points.append(PointStruct(
                id=i,
                vector=v.tolist() if hasattr(v, "tolist") else list(v),
                payload=payload
            ))

        self.client.upsert(collection_name=collection, points=points)

    def search(self, query: str, top_k: int = DENSE_TOP_K, collection: str = COLLECTION_NAME) -> list[SearchResult]:
        """Search using dense vectors."""
        if not query.strip():
            return []

        query_vector = self._get_encoder().encode(query, show_progress_bar=False)
        vec_list = query_vector.tolist() if hasattr(query_vector, "tolist") else list(query_vector)

        response = self.client.query_points(
            collection_name=collection,
            query=vec_list,
            limit=top_k
        )

        results = []
        for pt in response.points:
            payload = pt.payload or {}
            text = payload.get("text", "")
            meta = {k: v for k, v in payload.items() if k != "text"}
            results.append(SearchResult(
                text=text,
                score=float(pt.score) if pt.score is not None else 0.0,
                metadata=meta,
                method="dense"
            ))
        return results


def reciprocal_rank_fusion(results_list: list[list[SearchResult]], k: int = 60,
                           top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
    """Merge ranked lists using RRF: score(d) = Σ 1/(k + rank)."""
    rrf_scores: dict[str, dict] = {}
    for result_list in results_list:
        for rank, result in enumerate(result_list):
            doc_key = result.text
            if doc_key not in rrf_scores:
                rrf_scores[doc_key] = {
                    "score": 0.0,
                    "metadata": dict(result.metadata),
                    "text": result.text
                }
            rrf_scores[doc_key]["score"] += 1.0 / (k + rank + 1)
            if result.metadata:
                rrf_scores[doc_key]["metadata"].update(result.metadata)

    sorted_docs = sorted(rrf_scores.values(), key=lambda x: x["score"], reverse=True)[:top_k]

    return [
        SearchResult(
            text=item["text"],
            score=item["score"],
            metadata=item["metadata"],
            method="hybrid"
        )
        for item in sorted_docs
    ]


class HybridSearch:
    """Combines BM25 + Dense + RRF. (Đã implement sẵn — dùng classes ở trên)"""
    def __init__(self):
        self.bm25 = BM25Search()
        self.dense = DenseSearch()

    def index(self, chunks: list[dict]) -> None:
        self.bm25.index(chunks)
        self.dense.index(chunks)

    def search(self, query: str, top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
        bm25_results = self.bm25.search(query, top_k=BM25_TOP_K)
        dense_results = self.dense.search(query, top_k=DENSE_TOP_K)
        return reciprocal_rank_fusion([bm25_results, dense_results], top_k=top_k)


if __name__ == "__main__":
    print(f"Original:  Nhân viên được nghỉ phép năm")
    print(f"Segmented: {segment_vietnamese('Nhân viên được nghỉ phép năm')}")
