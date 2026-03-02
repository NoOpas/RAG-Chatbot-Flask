import psycopg2
from sentence_transformers import SentenceTransformer
from langchain_text_splitter import RecursiveCharacterTextSplitter
from loguru import logger 
import sys # Для логгера

# === CONFIGURATION ===
DB_CONFIG = {
    "database": "my_db",
    "user": "my_user",
    "password": "my_password",
    "host": "localhost"
}

MODEL_PATH = "./models/multilingual-e5-small"
BATCH_SIZE = 32
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# === настройка LOGURU ===
logger.remove()

logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
    colorize=True
)

# добавляем логи (сохраняет последние 3 лога, до 100 MB каждый)
logger.add(
    "logs/vectorization_{time:YYYY-MM-DD}.log",
    rotation="100 MB",
    retention=3,
    level="DEBUG",
    encoding="utf-8"
)

# === Загрузка модели ===
logger.info("Загрузка модели эмбеддингов...")
model = SentenceTransformer(MODEL_PATH, device="cpu")
logger.success("✅ Модель успешно загружена.")

# === Создает делитель текста ===
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
    length_function=len,
)

# === Подключение к базе ===
def get_db_connection():
    return psycopg2.connect(**DB_CONFIG)

conn = get_db_connection()
cursor = conn.cursor()

# === 4. Сбор всех записей для векторизации ===
logger.info("Получение записей из БД для векторизации...")
cursor.execute("""
    SELECT id, url, content 
    FROM main_table
    WHERE content IS NOT NULL AND content != ''
""")

rows = cursor.fetchall()
logger.info(f"Найдено {len(rows)} записей для обработки.")

# === Создание таблицы document_chunks, если её нет ===
logger.info("Проверка/создание таблицы main_table_chunks...")
cursor.execute("""
    CREATE TABLE IF NOT EXISTS main_table_chunks (
        id SERIAL PRIMARY KEY,
        original_id INT REFERENCES main_table(id) ON DELETE CASCADE,
        url TEXT,
        content TEXT,
        embedding VECTOR(384),
        created_at TIMESTAMPTZ DEFAULT NOW()
    );
""")
conn.commit()
logger.success("✅ Таблица main_table_chunks готова.")

# === Запись -> Чанк -> Эмбеддинг -> Вставка в базу ===
processed_count = 0

for idx, (row_id, url, content) in enumerate(rows, start=1):
    try:
        # Разбивка на куски
        chunks = text_splitter.split_text(content)
        logger.debug(f"ID {row_id}: получено {len(chunks)} чанков.")

        if not chunks:
            logger.warning(f"ID {row_id}: пустые чанки после разбиения — пропуск.")
            continue

        # Создание префикса для E5
        prefixed_chunks = ["passage: " + chunk for chunk in chunks]

        # Генерация эмбеддингов
        embeddings = model.encode(
            prefixed_chunks,
            batch_size=BATCH_SIZE,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        # Вставка нового куска в базу как новую запись
        for chunk_text, embedding_vector in zip(chunks, embeddings):
            cursor.execute("""
                INSERT INTO main_table_chunks (original_id, url, content, embedding)
                VALUES (%s, %s, %s, %s)
            """, (row_id, url, chunk_text, embedding_vector.tolist()))

        processed_count += 1

        # Фиксирование каждых 10 записей
        if processed_count % 10 == 0:
            conn.commit()
            logger.info(f"💾 Зафиксировано {processed_count} из {len(rows)} документов. ({idx}/{len(rows)})")

    except Exception as e:
        logger.error(f"❌ Ошибка при обработке ID {row_id}: {e}")
        logger.exception(e)  # Логаем весь ответ ошибки
        continue

# Последняя фиксация обработки документов
conn.commit()
logger.success(f"🏁 Успешно обработано {processed_count} документов.")

# === Создание HNSW (алгоритм приблизительного поиска ближайших соседей) индекса ===
logger.info("Создание индекса HNSW для быстрого поиска по эмбеддингам...")
try:
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS hnsw_embedding_idx 
        ON main_table_chunks 
        USING hnsw (embedding vector_cosine_ops);
    """)
    conn.commit()
    logger.success("✅ Индекс hnsw_embedding_idx создан.")
except Exception as e:
    logger.error(f"⚠️ Не удалось создать индекс: {e}")
    logger.exception(e)

# === Завершение ===
cursor.close()
conn.close()
logger.info("🔌 Соединение с базой данных закрыто.")

# Вывод при окончании работы программы
logger.info("="*60)
logger.info("📊 ВЕКТОРИЗАЦИЯ ЗАВЕРШЕНА")
logger.info(f"✅ Обработано документов: {processed_count}")
logger.info(f"📂 Логи сохранены в: logs/vectorization_*.log")
logger.info("="*60)
