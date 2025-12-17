# app/db/connection.py
import psycopg2
from app.logging import logger
from app.settings import Settings

_db_conn = None

def get_db_connection():
    logger.info("Установка соединения с базой данных...")
    return psycopg2.connect(
        database=Settings.DB_NAME,
        user=Settings.DB_USER,
        password=Settings.DB_PASSWORD,
        host=Settings.DB_HOST
    )

def get_persistent_db_connection():
    global _db_conn
    if _db_conn is None or _db_conn.closed:
        _db_conn = get_db_connection()
        logger.success("✅ Соединение с базой данных установлено.")
    return _db_conn
