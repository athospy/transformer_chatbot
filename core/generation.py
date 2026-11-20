"""Generation: answer a question with a fine-tuned DialoGPT-small.

Used by the website (the "What the fine-tuned model says" button) and by the eval, so both
generate with exactly the same settings.
"""
from __future__ import annotations

import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 72 new tokens fit 9 in 10 dataset answers with their link (median 42, longest 96). A model that
# learned to end its answer stops earlier; the 2024 model never does, so it always uses all 72.
# The first version (5 beams, max_length=100) took 15-45 s on the 1-CPU server and hit timeouts.
GENERATION_SETTINGS = {
    'max_new_tokens': 72,          # cap the answer length (not counting the question)
    'num_beams': 3,                # keep the 3 best partial answers at each step
    'early_stopping': True,        # stop when every beam has finished its answer
    'no_repeat_ngram_size': 3,     # never repeat the same 3 words in a row
    'repetition_penalty': 1.2,     # discourage repeating words the model already used
}


def clean_answer(text):
    """Trim spaces and the stray leading '.' the 2024 model tends to start with."""
    return text.strip().lstrip('.').strip()


class Generator:
    def __init__(self, model_path, tokenizer_path=None, settings=None):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path or model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path).eval()
        self.settings = settings or GENERATION_SETTINGS

    @torch.no_grad()
    def answer(self, question):
        """Returns (answer text, milliseconds spent generating)."""
        # DialoGPT format: the user's turn ends with the end-of-sequence token, then the reply
        inputs = self.tokenizer.encode(question + self.tokenizer.eos_token, return_tensors='pt')
        started = time.perf_counter()
        output = self.model.generate(
            inputs,
            attention_mask=torch.ones(inputs.shape),        # no padding: every token is real
            pad_token_id=self.tokenizer.eos_token_id,       # GPT-2 has no pad token, so pad with eos
            **self.settings,
        )
        elapsed_ms = round((time.perf_counter() - started) * 1000)
        # Decode only the new tokens, so the answer doesn't start with the question
        return clean_answer(self.tokenizer.decode(output[0][inputs.shape[1]:], skip_special_tokens=True)), elapsed_ms
