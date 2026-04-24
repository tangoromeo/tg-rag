from sentence_transformers import CrossEncoder

import config

RERANK_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


class Reranker:
    def __init__(self) -> None:
        print(f"Загрузка reranker {RERANK_MODEL}...", flush=True)
        self.model = CrossEncoder(RERANK_MODEL, device=config.EMBED_DEVICE)

    def rerank(self, query: str, hits: list, top_k: int = 5) -> list[tuple[float, object]]:
        pairs = [(query, hit.payload.get("text", "")) for hit in hits]
        scores = self.model.predict(pairs)
        ranked = sorted(zip(scores, hits), key=lambda x: x[0], reverse=True)
        return ranked[:top_k]
