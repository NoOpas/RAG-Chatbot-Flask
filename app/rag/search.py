# app/rag/search.py
from app.db.connection import get_persistent_db_connection
from app.models.embedding import EmbeddingModel
from app.logging import logger

def search_similar_texts(query: str, top_k: int = 3):
    logger.info(f"🔍 Поиск похожих текстов для запроса: '{query[:60]}...'")
    embedding_model = EmbeddingModel()
    query_embedding = embedding_model.encode(query)

    conn = get_persistent_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, url, content, original_id
        FROM document_chunks
        ORDER BY embedding <=> %s::vector
        LIMIT %s * 5
    """, (query_embedding, top_k))

    raw_results = cursor.fetchall()
    cursor.close()
    
    logger.info(f"📥 Получено {len(raw_results)} чанков из БД. IDs: {[r[0] for r in raw_results]}")
    return raw_results
