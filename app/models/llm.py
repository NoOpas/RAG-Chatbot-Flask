# app/models/llm.py
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
    GenerationConfig
)
from threading import Thread, Lock
from app.settings import Settings

class LLMModel:
    _instance = None
    _lock = Lock()  # For thread safety

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                # Double-check inside lock
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        from app.logging import logger
        logger.info("Загрузка Saiga-Mistral-7B-GPTQ...")

        self.device_map = "auto" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(
            Settings.MODEL_PATH,
            device_map=self.device_map,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(Settings.MODEL_PATH)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.generation_config = GenerationConfig(
            max_new_tokens=Settings.MAX_NEW_TOKENS,
            temperature=0.2,
            top_p=0.9,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )

        logger.success("✅ Saiga-Mistral-7B-GPTQ загружена.")

    def stream_generate(self, inputs, stopping_criteria):
        from app.logging import logger
        logger.info("Начало генерации ответа...")

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )

        gen_kwargs = {
            "input_ids": inputs["input_ids"].to(self.model.device),
            "attention_mask": inputs["attention_mask"].to(self.model.device),
            "generation_config": self.generation_config,
            "stopping_criteria": stopping_criteria,
            "streamer": streamer,
        }

        thread = Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.start()
        yield from streamer
        logger.info("Генерация завершена")
