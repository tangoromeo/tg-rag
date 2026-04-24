import hashlib
import json
import sqlite3
import uuid
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

import config
from chunker import chunk_messages

DB_PATH = Path(config.DB_PATH)


def get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS indexed_chunks (
            point_id    TEXT PRIMARY KEY,
            chat_id     TEXT,
            message_ids TEXT
        )
    """)
    conn.commit()
    return conn


def get_qdrant() -> QdrantClient:
    return QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)


def ensure_collection(client: QdrantClient) -> None:
    existing = {c.name for c in client.get_collections().collections}
    if config.COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=config.COLLECTION_NAME,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
        )
        print(f"Коллекция '{config.COLLECTION_NAME}' создана.")


def make_point_id(chat_id: str, message_ids: list[int]) -> str:
    key = f"{chat_id}:{sorted(message_ids)}"
    return str(uuid.UUID(hashlib.md5(key.encode()).hexdigest()))


def run_indexer() -> None:
    conn = get_conn()
    client = get_qdrant()
    ensure_collection(client)

    print(f"Загрузка модели {config.EMBED_MODEL} на устройство {config.EMBED_DEVICE}...")
    model = SentenceTransformer(config.EMBED_MODEL, device=config.EMBED_DEVICE)

    chat_ids = [
        row[0]
        for row in conn.execute("SELECT DISTINCT chat_id FROM messages").fetchall()
    ]
    if not chat_ids:
        print("Нет сообщений в базе данных.")
        return

    total_indexed = 0

    for chat_id in chat_ids:
        print(f"\nОбработка чата {chat_id}...")
        rows = conn.execute(
            """
            SELECT id, chat_id, ts, text, from_id, from_name, reply_to_id
            FROM messages
            WHERE chat_id = ?
            """,
            (chat_id,),
        ).fetchall()

        messages = [
            {
                "id": r[0],
                "chat_id": r[1],
                "ts": r[2],
                "text": r[3],
                "from_id": r[4],
                "from_name": r[5],
                "reply_to_id": r[6],
            }
            for r in rows
        ]

        chunks = chunk_messages(messages)
        print(f"  Всего чанков: {len(chunks)}")

        new_chunks: list[tuple[str, dict]] = []
        for chunk in chunks:
            point_id = make_point_id(chunk["chat_id"], chunk["message_ids"])
            exists = conn.execute(
                "SELECT 1 FROM indexed_chunks WHERE point_id = ?", (point_id,)
            ).fetchone()
            if not exists:
                new_chunks.append((point_id, chunk))

        if not new_chunks:
            print("  Нет новых чанков для индексации.")
            continue

        print(f"  Новых чанков: {len(new_chunks)}")
        texts = ["passage: " + c["text"] for _, c in new_chunks]

        batch_size = 64
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            vecs = model.encode(batch, show_progress_bar=False, normalize_embeddings=True)
            all_embeddings.extend(vecs)
            done = min(i + batch_size, len(texts))
            print(f"  Эмбеддинги: {done}/{len(texts)}", flush=True)

        points = [
            PointStruct(
                id=point_id,
                vector=vec.tolist(),
                payload={
                    "chat_id": chunk["chat_id"],
                    "ts_start": chunk["ts_start"],
                    "ts_end": chunk["ts_end"],
                    "message_ids": chunk["message_ids"],
                    "authors": chunk["authors"],
                    "text": chunk["text"],
                },
            )
            for (point_id, chunk), vec in zip(new_chunks, all_embeddings)
        ]

        client.upsert(collection_name=config.COLLECTION_NAME, points=points)

        for point_id, chunk in new_chunks:
            conn.execute(
                "INSERT OR REPLACE INTO indexed_chunks (point_id, chat_id, message_ids) VALUES (?, ?, ?)",
                (point_id, chunk["chat_id"], json.dumps(chunk["message_ids"])),
            )
        conn.commit()
        total_indexed += len(points)
        print(f"  Проиндексировано: {len(points)} чанков.")

    print(f"\nИндексация завершена. Всего проиндексировано: {total_indexed} чанков.")


if __name__ == "__main__":
    run_indexer()
