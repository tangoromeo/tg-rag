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

VECTOR_CANDIDATES = 20

DB_PATH = Path(config.DB_PATH)


def resolve_chat_id(chat_arg: str) -> str:
    """Resolve username or title to chat_id using local chats table."""
    if not DB_PATH.exists():
        return chat_arg
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute(
        "SELECT chat_id FROM chats WHERE chat_id = ? OR username = ? OR title = ? LIMIT 1",
        (chat_arg, chat_arg, chat_arg),
    ).fetchone()
    conn.close()
    return row[0] if row else chat_arg


def run_search(
    query: str,
    chat_arg: str | None = None,
    days: int | None = None,
    top_k: int = 5,
) -> None:
    print(f"Загрузка модели {config.EMBED_MODEL}...", flush=True)
    model = SentenceTransformer(config.EMBED_MODEL, device=config.EMBED_DEVICE)
    reranker = Reranker()
    client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)

    vec = model.encode(["query: " + query], normalize_embeddings=True)[0].tolist()

    conditions = []
    if chat_arg:
        chat_id = resolve_chat_id(chat_arg)
        conditions.append(FieldCondition(key="chat_id", match=MatchValue(value=chat_id)))
    if days:
        ts_min = int(time.time()) - days * 86400
        conditions.append(FieldCondition(key="ts_start", range=Range(gte=ts_min)))

    search_filter = Filter(must=conditions) if conditions else None

    candidates = client.query_points(
        collection_name=config.COLLECTION_NAME,
        query=vec,
        limit=VECTOR_CANDIDATES,
        query_filter=search_filter,
        with_payload=True,
    ).points

    if not candidates:
        print("Результатов не найдено.")
        return

    print(f"Кандидатов от векторного поиска: {len(candidates)}. Rerankинг...", flush=True)
    results = reranker.rerank(query, candidates, top_k=top_k)

    print(f"\nТоп-{len(results)} после rerank:\n")
    for i, (score, hit) in enumerate(results, 1):
        p = hit.payload
        date_str = datetime.fromtimestamp(p.get("ts_start", 0)).strftime("%Y-%m-%d %H:%M")
        chat = p.get("chat_id", "—")
        text = p.get("text", "")
        authors = ", ".join(p.get("authors", [])) or "—"

        print("─" * 60)
        print(f"[{i}] Rerank: {score:.4f} | Чат: {chat} | Дата: {date_str}")
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
    args = parser.parse_args()

    query = args.query
    if not query:
        print("Введите запрос: ", end="", flush=True)
        query = sys.stdin.readline().strip()

    if not query:
        print("Запрос не задан.", file=sys.stderr)
        sys.exit(1)

    run_search(query, chat_arg=args.chat, days=args.days, top_k=args.top_k)


if __name__ == "__main__":
    main()
