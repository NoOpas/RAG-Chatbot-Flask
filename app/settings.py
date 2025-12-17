# app/settings.py
import os

class Settings():
    # Paths
    MODEL_PATH = "./models/saiga_mistral_7b-GPTQ"
    EMBEDDING_MODEL_PATH = "./models/multilingual-e5-small"
    
    # DB
    DB_HOST = "******"
    DB_NAME = "******"
    DB_USER = "******"
    DB_PASSWORD = "******"
    
    # RAG
    TOP_K = 3
    MAX_CONTEXT_TOKENS = 3000
    MAX_NEW_TOKENS = 300
    
    # Logging
    LOG_FILE = "./logs/rag_chat.log"
