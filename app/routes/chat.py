# app/routes/chat.py
from flask import render_template, request, Response
from app.rag.pipeline import rag_pipeline_stream
import json

def init_routes(app):
    
    @app.route("/")
    def index():
        return render_template("index_streaming.html")

    @app.route("/stream_response")
    def stream_response():
        query = request.args.get("message", "").strip()
        if not query:
            return Response(
                "data: " + json.dumps({"type": "error", "text": "Пустой запрос"}) + "\n\n",
                content_type="text/event-stream"
            )

        def safe_stream():
            try:
                yield from rag_pipeline_stream(query)
            except GeneratorExit:
                pass
            except Exception as e:
                import logging
                logging.exception("SSE stream error")
                yield "data: " + json.dumps({
                    "type": "error",
                    "text": "Внутренняя ошибка сервера."
                }) + "\n\n"

        return Response(safe_stream(), content_type="text/event-stream")