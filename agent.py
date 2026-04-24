import json
import sqlite3
from datetime import datetime
from pathlib import Path

import ollama

import config
from search import search_chunks

SYSTEM_PROMPT = """\
Ты — ассистент, который отвечает на вопросы по истории Telegram-чатов.
Отвечай ТОЛЬКО на русском языке. Никогда не используй китайский, английский или любой другой язык.

ОБЯЗАТЕЛЬНЫЕ ПРАВИЛА:
1. Ты НИКОГДА не отвечаешь на вопрос без предварительного вызова инструмента search.
2. Перед любым ответом вызови search хотя бы один раз.
3. Если первый поиск не дал достаточно информации — сделай ещё 1-2 запроса с другими формулировками.
4. Только после поиска формулируй ответ на русском языке. Будь конкретным.
5. Если после поиска информации всё равно недостаточно — скажи об этом прямо.\
"""

TOOL_DEF = {
    "type": "function",
    "function": {
        "name": "search",
        "description": "Поиск по базе сообщений из Telegram чатов",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Поисковый запрос на русском языке",
                }
            },
            "required": ["query"],
        },
    },
}

MAX_TOOL_CALLS = 5


def _chat_title(chat_id: str) -> str:
    db = Path(config.DB_PATH)
    if not db.exists():
        return chat_id
    conn = sqlite3.connect(db)
    row = conn.execute(
        "SELECT title, username FROM chats WHERE chat_id = ?", (chat_id,)
    ).fetchone()
    conn.close()
    if row:
        return row[0] or row[1] or chat_id
    return chat_id


def _run_search(query: str) -> list[dict]:
    results = search_chunks(query, top_k=5)
    return [
        {
            "text": r["text"],
            "chat_name": _chat_title(r["chat_id"]),
            "date": datetime.fromtimestamp(r["ts_start"]).strftime("%Y-%m-%d"),
            "score": round(r["score"], 4),
        }
        for r in results
    ]


class Agent:
    def __init__(self) -> None:
        self.client = ollama.Client(host=config.OLLAMA_HOST, timeout=120.0)
        self.model = config.OLLAMA_MODEL

    def chat(self, question: str) -> str:
        try:
            return self._tool_chat(question)
        except Exception as e:
            print(f"Tool use недоступен ({e}), переключаюсь на fallback...", flush=True)
            return self._fallback_chat(question)

    def _tool_chat(self, question: str) -> str:
        messages: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        tool_calls_made = 0

        while tool_calls_made < MAX_TOOL_CALLS:
            response = self.client.chat(
                model=self.model,
                messages=messages,
                tools=[TOOL_DEF],
            )
            msg = response.message

            # If model skipped tool use on the first turn, force a search with the raw question
            if not msg.tool_calls and tool_calls_made == 0:
                print(f"  [поиск] {question!r}", flush=True)
                results = _run_search(question)
                tool_calls_made += 1
                messages.append({"role": "assistant", "content": ""})
                messages.append({
                    "role": "user",
                    "content": (
                        "Вот результаты поиска по твоему вопросу:\n"
                        + json.dumps(results, ensure_ascii=False, indent=2)
                        + "\n\nТеперь дай развёрнутый ответ на основе этих данных."
                    ),
                })
                continue

            # Don't echo tool_calls back into history — Ollama SDK serializes
            # arguments inconsistently across versions (dict vs JSON string).
            # Inject results as a user message instead; model handles it fine.
            messages.append({"role": "assistant", "content": msg.content or ""})

            if not msg.tool_calls:
                return msg.content or ""

            for tc in msg.tool_calls:
                if tc.function.name != "search":
                    continue
                args = tc.function.arguments
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                query = args.get("query", "") if isinstance(args, dict) else ""
                print(f"  [поиск] {query!r}", flush=True)
                results = _run_search(query)
                tool_calls_made += 1
                messages.append({
                    "role": "user",
                    "content": (
                        f"Результаты поиска по «{query}»:\n"
                        + json.dumps(results, ensure_ascii=False, indent=2)
                        + "\n\nЕсли нужно — сделай ещё запросы, иначе дай ответ."
                    ),
                })

        # Max tool calls reached — force synthesis
        messages.append({
            "role": "user",
            "content": "Подведи итог на основе найденной информации.",
        })
        response = self.client.chat(model=self.model, messages=messages)
        return response.message.content or ""

    def _fallback_chat(self, question: str) -> str:
        """JSON-based fallback when the model doesn't support tool calling."""
        messages: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"{question}\n\n"
                    "Сначала верни только JSON без пояснений: "
                    '{"queries": ["запрос1", "запрос2"]} — '
                    "список поисковых запросов для ответа на этот вопрос."
                ),
            },
        ]
        response = self.client.chat(model=self.model, messages=messages)
        raw = response.message.content or ""

        try:
            start, end = raw.index("{"), raw.rindex("}") + 1
            queries: list[str] = json.loads(raw[start:end]).get("queries", [])
        except (ValueError, json.JSONDecodeError):
            queries = [question]

        all_results: list[dict] = []
        for q in queries[:MAX_TOOL_CALLS]:
            print(f"  [поиск] {q!r}", flush=True)
            all_results.extend(_run_search(q))

        context = json.dumps(all_results, ensure_ascii=False, indent=2)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
            {"role": "assistant", "content": f"Найденные фрагменты:\n{context}"},
            {"role": "user", "content": "Дай развёрнутый ответ на основе найденной информации."},
        ]
        response = self.client.chat(model=self.model, messages=messages)
        return response.message.content or ""
