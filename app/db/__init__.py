# app/db/__init__.py
from .connection import get_persistent_db_connection

__all__ = ["get_persistent_db_connection"]
