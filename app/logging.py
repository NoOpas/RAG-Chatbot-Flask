# app/logging.py
import os
import sys
from loguru import logger
from app.settings import Settings


# Ensure logs dir exists
os.makedirs("./logs", exist_ok=True)

# Remove default handler
logger.remove()

# Console handler (stdout, colored, thread-safe)
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
    colorize=True,
    enqueue=True,   # ← critical for threads/SSE
    catch=True
)

# File handler (rotating, debug-level)
logger.add(
    Settings.LOG_FILE,
    rotation="100 MB",
    retention=3,
    level="DEBUG",
    encoding="utf-8",
    enqueue=True, 
    catch=True
)

# Export for import elsewhere
__all__ = ["logger"]
