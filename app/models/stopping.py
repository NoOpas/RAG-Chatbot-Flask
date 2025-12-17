# app/models/stopping.py
from transformers import StoppingCriteria

class StopOnSequence(StoppingCriteria):
    def __init__(self, stop_sequence: str, tokenizer):
        super().__init__()
        self.stop_sequence = stop_sequence
        self.stop_ids = tokenizer.encode(stop_sequence, add_special_tokens=False)
        self.stop_len = len(self.stop_ids)

    def __call__(self, input_ids, scores, **kwargs):
        for seq in input_ids:
            if len(seq) >= self.stop_len and seq[-self.stop_len:].tolist() == self.stop_ids:
                return True
        return False
