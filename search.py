import sqlite3
import sys
import time
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue, Range
from sentence_transformers import SentenceTransformer

import config
from reranker import Reranker

DB_PATH = Path(config.DB_PATH)
VECTOR_CANDIDATES = 20

# Lazy singletons — loaded once, reused across calls (including by agent.py)
_model: SentenceTransformer | None = None
_reranker: Reranker | None = None
_qdrant: QdrantClient | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print(f"Загрузка модели {config.EMBED_MODEL}...", flush=True)
        _model = SentenceTransformer(config.EMBED_MODEL, device=config.EMBED_DEVICE)
    return _model


def _get_reranker() -> Reranker:
    global _reranker
    if _reranker is None:
        _reranker = Reranker()
    return _reranker


def _get_qdrant() -> QdrantClient:
    global _qdrant
    if _qdrant is None:
        _qdrant = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)
    return _qdrant


def resolve_chat_id(chat_arg: str) -> str:
    if not DB_PATH.exists():
        return chat_arg
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute(
        "SELECT chat_id FROM chats WHERE chat_id = ? OR username = ? OR title = ? LIMIT 1",
        (chat_arg, chat_arg, chat_arg),
    ).fetchone()
    conn.close()
    return row[0] if row else chat_arg


def search_chunks(
    query: str,
    chat_arg: str | None = None,
    days: int | None = None,
    top_k: int = 5,
) -> list[dict]:
    """Embed → Qdrant top-20 → rerank top-k. Returns list of result dicts."""
    vec = _get_model().encode(["query: " + query], normalize_embeddings=True)[0].tolist()

    conditions = []
    if chat_arg:
        chat_id = resolve_chat_id(chat_arg)
        conditions.append(FieldCondition(key="chat_id", match=MatchValue(value=chat_id)))
    if days:
        ts_min = int(time.time()) - days * 86400
        conditions.append(FieldCondition(key="ts_start", range=Range(gte=ts_min)))

    search_filter = Filter(must=conditions) if conditions else None

    candidates = _get_qdrant().query_points(
        collection_name=config.COLLECTION_NAME,
        query=vec,
        limit=VECTOR_CANDIDATES,
        query_filter=search_filter,
        with_payload=True,
    ).points

    if not candidates:
        return []

    ranked = _get_reranker().rerank(query, candidates, top_k=top_k)
    return [
        {
            "score": float(score),
            "text": hit.payload.get("text", ""),
            "chat_id": hit.payload.get("chat_id", ""),
            "ts_start": hit.payload.get("ts_start", 0),
            "authors": hit.payload.get("authors", []),
        }
        for score, hit in ranked
    ]


def run_search(
    query: str,
    chat_arg: str | None = None,
    days: int | None = None,
    top_k: int = 5,
) -> None:
    results = search_chunks(query, chat_arg=chat_arg, days=days, top_k=top_k)

    if not results:
        print("Результатов не найдено.")
        return

    print(f"\nТоп-{len(results)} после rerank:\n")
    for i, r in enumerate(results, 1):
        date_str = datetime.fromtimestamp(r["ts_start"]).strftime("%Y-%m-%d %H:%M")
        authors = ", ".join(r["authors"]) or "—"
        text = r["text"]

        print("─" * 60)
        print(f"[{i}] Rerank: {r['score']:.4f} | Чат: {r['chat_id']} | Дата: {date_str}")
        print(f"Авторы: {authors}")
        print()
        print(text[:800] + ("…" if len(text) > 800 else ""))
        print()

    print("─" * 60)


def main() -> None:
    parser = ArgumentParser(description="Поиск по индексу Telegram-чатов")
    parser.add_argument("query", nargs="?", help="поисковый запрос")
    parser.add_argument("--chat", help="фильтр по chat_id, username или названию чата")
    parser.add_argument("--days", type=int, help="фильтр: только за последние N дней")
    parser.add_argument("--top-k", type=int, default=5, dest="top_k")
    parser.add_argument("--agent", action="store_true", help="режим агента с синтезом ответа через Ollama")
    args = parser.parse_args()

    query = args.query
    if not query:
        print("Введите запрос: ", end="", flush=True)
        query = sys.stdin.readline().strip()

    if not query:
        print("Запрос не задан.", file=sys.stderr)
        sys.exit(1)

    if args.agent:
        from agent import Agent
        answer = Agent().chat(query)
        print("\n" + "─" * 60)
        print(answer)
        print("─" * 60)
    else:
        run_search(query, chat_arg=args.chat, days=args.days, top_k=args.top_k)


if __name__ == "__main__":
    main()
