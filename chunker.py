from collections import defaultdict
from typing import TypedDict

import config


class Chunk(TypedDict):
    text: str
    chat_id: str
    ts_start: int
    ts_end: int
    message_ids: list[int]
    authors: list[str]


def chunk_messages(messages: list[dict]) -> list[Chunk]:
    if not messages:
        return []

    chat_id = messages[0]["chat_id"]
    by_id: dict[int, dict] = {m["id"]: m for m in messages}
    children: dict[int, list[int]] = defaultdict(list)
    roots: list[int] = []

    for msg in messages:
        reply_to = msg["reply_to_id"]
        if reply_to and reply_to in by_id:
            children[reply_to].append(msg["id"])
        else:
            roots.append(msg["id"])

    chunks: list[Chunk] = []
    orphan_roots: list[dict] = []

    for root_id in roots:
        has_replies = bool(children.get(root_id))
        if not has_replies:
            orphan_roots.append(by_id[root_id])
            continue

        total = _subtree_size(root_id, children)
        if total <= config.CHUNK_MAX_MESSAGES:
            chunks.append(_make_chunk(_collect_subtree(root_id, children), by_id, chat_id))
        else:
            chunks.extend(_split_tree(root_id, children, by_id, chat_id))

    chunks.extend(_chunk_orphans(orphan_roots, chat_id))
    return chunks


def _collect_subtree(root_id: int, children: dict) -> list[int]:
    result: list[int] = []
    stack = [root_id]
    while stack:
        node = stack.pop()
        result.append(node)
        for child in reversed(children.get(node, [])):
            stack.append(child)
    return result


def _subtree_size(root_id: int, children: dict) -> int:
    return len(_collect_subtree(root_id, children))


def _split_tree(root_id: int, children: dict, by_id: dict, chat_id: str) -> list[Chunk]:
    chunks: list[Chunk] = []

    def split(node_id: int) -> None:
        if _subtree_size(node_id, children) <= config.CHUNK_MAX_MESSAGES:
            chunks.append(_make_chunk(_collect_subtree(node_id, children), by_id, chat_id))
            return

        current: list[int] = [node_id]
        for child_id in children.get(node_id, []):
            child_size = _subtree_size(child_id, children)
            if child_size <= config.CHUNK_MAX_MESSAGES:
                child_sub = _collect_subtree(child_id, children)
                if len(current) + len(child_sub) <= config.CHUNK_MAX_MESSAGES:
                    current.extend(child_sub)
                else:
                    chunks.append(_make_chunk(child_sub, by_id, chat_id))
            else:
                split(child_id)

        if current:
            chunks.append(_make_chunk(current, by_id, chat_id))

    split(root_id)
    return chunks


def _make_chunk(msg_ids: list[int], by_id: dict, chat_id: str) -> Chunk:
    msgs = sorted((by_id[mid] for mid in msg_ids if mid in by_id), key=lambda m: m["ts"])
    authors = list({m["from_name"] or m["from_id"] for m in msgs if m.get("from_name") or m.get("from_id")})
    return Chunk(
        text="\n".join(m["text"] for m in msgs),
        chat_id=chat_id,
        ts_start=msgs[0]["ts"],
        ts_end=msgs[-1]["ts"],
        message_ids=[m["id"] for m in msgs],
        authors=authors,
    )


def _chunk_orphans(orphans: list[dict], chat_id: str) -> list[Chunk]:
    if not orphans:
        return []

    orphans = sorted(orphans, key=lambda m: m["ts"])
    window_sec = config.ORPHAN_WINDOW_MINUTES * 60
    chunks: list[Chunk] = []
    group: list[dict] = [orphans[0]]

    def flush(g: list[dict]) -> None:
        id_map = {m["id"]: m for m in g}
        chunks.append(_make_chunk([m["id"] for m in g], id_map, chat_id))

    for msg in orphans[1:]:
        if (
            msg["ts"] - group[-1]["ts"] <= window_sec
            and len(group) < config.CHUNK_MAX_MESSAGES
        ):
            group.append(msg)
        else:
            flush(group)
            group = [msg]

    flush(group)
    return chunks
