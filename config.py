import os
from dotenv import load_dotenv

load_dotenv()

CHUNK_MAX_MESSAGES = 16
ORPHAN_WINDOW_MINUTES = 5
COLLECTION_NAME = "tg_rag"
EMBED_MODEL = "intfloat/multilingual-e5-large"
EMBED_DEVICE = "mps"
DB_PATH = "data/raw.db"

TG_API_ID = int(os.getenv("TG_API_ID", "0"))
TG_API_HASH = os.getenv("TG_API_HASH", "")
TG_PHONE = os.getenv("TG_PHONE", "")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:14b")

RECENT_DAYS = 60          # кандидаты не старше N дней всегда попадают в пул
RECENT_CANDIDATES = 10    # сколько «свежих» кандидатов добавлять отдельным запросом
DECAY_HALF_LIFE_DAYS = 180  # через сколько дней релевантность падает вдвое
