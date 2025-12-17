# app/rag/pipeline.py
import json, torch, gc
from transformers import StoppingCriteriaList
from app.rag.search import search_similar_texts
from app.rag.context import truncate_context_by_tokens
from app.models.llm import LLMModel
from app.models.stopping import StopOnSequence
from app.logging import logger
from app.prompts.RAG_template import build_prompt


def rag_pipeline_stream(query: str):
    yield "data: " + json.dumps({"type": "status", "text": "Поиск информации..."}) + "\n\n"
    
    try:
        raw_results = search_similar_texts(query, top_k=3)
        if not raw_results:
            yield "data: " + json.dumps({"type": "error", "text": "Не найдено релевантных документов."}) + "\n\n"
            return

        # Group & merge (same logic as before)
        source_map = {}
        for row_id, url, content, orig_id in raw_results:
            logger.debug(f"📄 Чанк ID={row_id}, original_id={orig_id}, URL={url[:50]}...")
            if url not in source_map and len(source_map) < 3:
                source_map[url] = {"contents": [], "url": url}
            if url in source_map:
                source_map[url]["contents"].append(content)

        similar_texts = [
            (None, data["url"], "\n\n---\n\n".join(data["contents"][:2]))
            for data in source_map.values()
        ]

        # Build context
        raw_contents = [row[2] for row in similar_texts]
        context = truncate_context_by_tokens(raw_contents)
        sources = [row[1] for row in similar_texts]
        logger.info(f"Контекст подготовлен, длина: {len(context)} символов")

        # Generate
        llm = LLMModel()
        
        prompt = build_prompt(context=context, query=query)

        inputs = llm.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=6000
        )

        stop_criteria = StoppingCriteriaList([StopOnSequence("###", llm.tokenizer)])
        full_text = ""

        for new_text in llm.stream_generate(inputs, stop_criteria):
            full_text += new_text
            if "###" in full_text:
                clean_answer = full_text.split("###")[0].strip()
                logger.info(f"Сгенерирован ответ: '{clean_answer}'")
                yield "data: " + json.dumps({"type": "token", "text": clean_answer}) + "\n\n"
                break
            yield "data: " + json.dumps({"type": "token", "text": new_text}) + "\n\n"

        yield "data: " + json.dumps({
            "type": "token",
            "text": "\n\n"
        }) + "\n\n"

        yield "data: " + json.dumps({
            "type": "sources",
            "title": "📚 Источники:",
            "urls": sources
        }) + "\n\n"

    except Exception as e:
        yield "data: " + json.dumps({
            "type": "error",
            "text": f"Ошибка: {str(e)}"
        }) + "\n\n"

    finally:
        torch.cuda.empty_cache()
        gc.collect()
