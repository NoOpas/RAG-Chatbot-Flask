# chat.py
import sys

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

from app.logging import logger
logger.info("PYTORCH_ALLOC_CONF=expandable_segments:True")

from app import create_app
app = create_app()

if __name__ == "__main__":
    logger.info("🚀 Запуск приложения...")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
