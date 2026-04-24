import asyncio
import json
import sqlite3
import sys
from argparse import ArgumentParser
from pathlib import Path

from telethon import TelegramClient
from telethon.errors import FloodWaitError
from telethon.tl.types import Message

import config

DB_PATH = Path(config.DB_PATH)


def get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS messages (
            id          INTEGER,
            chat_id     TEXT,
            ts          INTEGER,
            text        TEXT,
            from_id     TEXT,
            from_name   TEXT,
            reply_to_id INTEGER,
            raw_json    TEXT,
            PRIMARY KEY (id, chat_id)
        );
        CREATE TABLE IF NOT EXISTS chats (
            chat_id  TEXT PRIMARY KEY,
            username TEXT,
            title    TEXT
        );
    """)
    conn.commit()
    return conn


def _build_client() -> TelegramClient:
    return TelegramClient("session", config.TG_API_ID, config.TG_API_HASH)


async def _resolve_sender(msg: Message, cache: dict[str, str]) -> tuple[str, str]:
    from_id = ""
    if msg.from_id:
        if hasattr(msg.from_id, "user_id"):
            from_id = str(msg.from_id.user_id)
        elif hasattr(msg.from_id, "channel_id"):
            from_id = str(msg.from_id.channel_id)

    if msg.post_author:
        return from_id, msg.post_author

    if not from_id:
        return "", ""

    if from_id in cache:
        return from_id, cache[from_id]

    try:
        sender = await msg.get_sender()
        if sender is None:
            name = ""
        elif hasattr(sender, "first_name"):
            parts = [sender.first_name or "", sender.last_name or ""]
            name = " ".join(p for p in parts if p).strip()
        elif hasattr(sender, "title"):
            name = sender.title or ""
        else:
            name = ""
    except Exception:
        name = ""

    cache[from_id] = name
    return from_id, name


async def _stream_messages(client: TelegramClient, entity, *, limit: int | None, min_id: int):
    sender_cache: dict[str, str] = {}
    retries = 0

    async def safe_iter():
        nonlocal retries
        while True:
            try:
                async for msg in client.iter_messages(
                    entity, limit=limit, min_id=min_id, reverse=True
                ):
                    retries = 0
                    yield msg
                return
            except FloodWaitError as e:
                wait = e.seconds + 1
                print(f"FloodWait: ждём {wait}с...", flush=True)
                await asyncio.sleep(wait)
            except Exception as e:
                retries += 1
                if retries > 5:
                    raise
                wait = 2**retries
                print(f"Ошибка: {e}. Повтор через {wait}с...", flush=True)
                await asyncio.sleep(wait)

    async for msg in safe_iter():
        if not isinstance(msg, Message):
            continue
        text = msg.message or ""
        if not text:
            continue
        from_id, from_name = await _resolve_sender(msg, sender_cache)
        yield msg, from_id, from_name


def _save_message(conn: sqlite3.Connection, chat_id: str, msg: Message, from_id: str, from_name: str) -> None:
    reply_to_id = msg.reply_to.reply_to_msg_id if msg.reply_to else None
    conn.execute(
        """
        INSERT OR REPLACE INTO messages (id, chat_id, ts, text, from_id, from_name, reply_to_id, raw_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            msg.id,
            chat_id,
            int(msg.date.timestamp()),
            msg.message or "",
            from_id,
            from_name,
            reply_to_id,
            json.dumps({"id": msg.id, "date": str(msg.date)}),
        ),
    )


def _save_chat(conn: sqlite3.Connection, chat_id: str, entity) -> None:
    username = getattr(entity, "username", None) or ""
    title = getattr(entity, "title", None) or getattr(entity, "first_name", None) or ""
    conn.execute(
        "INSERT OR REPLACE INTO chats (chat_id, username, title) VALUES (?, ?, ?)",
        (chat_id, username, title),
    )
    conn.commit()


async def fetch_chat(chat_id_or_username: str, limit: int | None = None) -> None:
    conn = get_conn()
    async with _build_client() as client:
        await client.start(phone=config.TG_PHONE)
        entity = await client.get_entity(chat_id_or_username)
        chat_id = str(entity.id)
        _save_chat(conn, chat_id, entity)

        count = 0
        async for msg, from_id, from_name in _stream_messages(client, entity, limit=limit, min_id=0):
            _save_message(conn, chat_id, msg, from_id, from_name)
            count += 1
            if count % 200 == 0:
                conn.commit()
                print(f"Загружено {count} сообщений...", flush=True)

        conn.commit()
        print(f"Готово. Всего загружено: {count} сообщений.")


async def fetch_new(chat_id_or_username: str) -> None:
    conn = get_conn()
    async with _build_client() as client:
        await client.start(phone=config.TG_PHONE)
        entity = await client.get_entity(chat_id_or_username)
        chat_id = str(entity.id)
        _save_chat(conn, chat_id, entity)

        row = conn.execute(
            "SELECT MAX(id) FROM messages WHERE chat_id = ?", (chat_id,)
        ).fetchone()
        min_id = row[0] if row[0] else 0
        print(f"Загрузка новых сообщений начиная с id={min_id}...")

        count = 0
        async for msg, from_id, from_name in _stream_messages(client, entity, limit=None, min_id=min_id):
            _save_message(conn, chat_id, msg, from_id, from_name)
            count += 1
            if count % 200 == 0:
                conn.commit()
                print(f"Загружено {count} новых сообщений...", flush=True)

        conn.commit()
        print(f"Готово. Загружено новых: {count} сообщений.")


def main() -> None:
    parser = ArgumentParser(description="Загрузка истории Telegram-чата")
    parser.add_argument("chat", help="username или id чата")
    parser.add_argument("--new", action="store_true", help="только новые сообщения")
    parser.add_argument("--limit", type=int, default=None, help="максимум сообщений")
    args = parser.parse_args()

    if args.new:
        asyncio.run(fetch_new(args.chat))
    else:
        asyncio.run(fetch_chat(args.chat, limit=args.limit))


if __name__ == "__main__":
    main()
