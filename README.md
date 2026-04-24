# tg-rag

Скачивает историю публичного Telegram-чата, разбивает на чанки по деревьям ответов, индексирует в Qdrant и ищет по смыслу через bi-encoder + cross-encoder reranker. LLM не используется.

## Архитектура

```
fetcher.py  →  SQLite (data/raw.db)  →  chunker.py  →  indexer.py  →  Qdrant
                                                                           ↓
                                                                       search.py
                                                                      reranker.py
```

| Шаг | Модель / инструмент |
|-----|---------------------|
| Эмбеддинги | `intfloat/multilingual-e5-large` (1024-dim, Cosine) |
| Reranker | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` |
| Векторная БД | Qdrant |
| Сырые данные | SQLite |
| Устройство | MPS (Apple Silicon) |

## Требования

- Python 3.11+
- [uv](https://github.com/astral-sh/uv)
- Docker (для Qdrant)
- Telegram API credentials: [my.telegram.org](https://my.telegram.org)

## Установка

```bash
git clone ...
cd tg-rag

cp .env .env  # уже создан, заполни значения
uv sync
```

## Конфигурация

`.env`:

```env
TG_API_ID=12345678
TG_API_HASH=abcdef...
TG_PHONE=+79991234567   # международный формат с +
QDRANT_HOST=localhost
QDRANT_PORT=6333
```

Параметры чанкинга и модели — в `config.py`.

## Запуск Qdrant

```bash
docker run -d -p 6333:6333 -v ./data/qdrant:/qdrant/storage qdrant/qdrant
```

## Использование

### 1. Скачать историю чата

```bash
# Полная загрузка
uv run python fetcher.py верхний_ларс

# Только новые сообщения (инкрементально)
uv run python fetcher.py верхний_ларс --new

# Ограничить количество
uv run python fetcher.py верхний_ларс --limit 5000
```

При первом запуске Telethon запросит код из SMS/приложения и создаст `session.session`.

### 2. Проиндексировать

```bash
uv run python indexer.py
```

Можно запускать параллельно с fetcher — индексирует уже скачанные сообщения, потом запустить ещё раз для доиндексации остатка.

### 3. Поиск

```bash
# Простой запрос
uv run python search.py "какие документы нужны для пересечения границы"

# С фильтрами
uv run python search.py "страховка" --chat верхний_ларс --days 30

# Из stdin
echo "виза в Грузию" | uv run python search.py
```

Векторный поиск возвращает 20 кандидатов, reranker отбирает топ-5.

## Структура проекта

```
tg-rag/
├── .env                  # credentials (в .gitignore)
├── config.py             # все константы
├── fetcher.py            # загрузка истории через Telethon
├── chunker.py            # разбивка на чанки по reply-деревьям
├── indexer.py            # эмбеддинги + загрузка в Qdrant
├── reranker.py           # cross-encoder reranker
├── search.py             # CLI поиска
└── data/
    ├── raw.db            # SQLite с сырыми сообщениями
    └── qdrant/           # хранилище Qdrant
```

## Логика чанкинга

1. **Reply-деревья**: сообщения группируются по `reply_to_id` рекурсивно.
2. **Маленькое дерево** (≤ 16 сообщений) → один чанк целиком.
3. **Большое дерево** (> 16) → рекурсивная нарезка по поддеревьям подходящего размера.
4. **Одиночки** (без ответов) → группируются в окна по 5 минут.

Текст чанка — сообщения в хронологическом порядке, без имён и меток времени.
