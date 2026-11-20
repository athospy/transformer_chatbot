"""Retrieval: answer a question with the closest written answer from the dataset.

Each dataset question is turned into an embedding (a vector of 384 numbers that captures its
meaning) with sentence-transformers/all-MiniLM-L6-v2. A visitor's question gets the same
treatment, and the closest dataset question (cosine similarity) wins. If even the closest one is
below `threshold`, the question is probably about something the data doesn't cover, so we say we
don't know instead of returning a wrong program. The threshold was chosen on the eval's dev split
(see eval/run_eval.py).

The model is used through plain `transformers` (mean pooling, like sentence-transformers does), so
no extra package is needed. Download it once:  python scripts/download_retriever.py
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

from core.faq_data import Entry, load_entries

RETRIEVER_PATH = Path(__file__).resolve().parent / 'retriever_model'
RETRIEVER_HUB_ID = 'sentence-transformers/all-MiniLM-L6-v2'

# Below this cosine similarity we answer "I don't know". Chosen on the eval's dev split.
DEFAULT_THRESHOLD = 0.56

# Every dataset question repeats the university's name, which makes unrelated questions look alike
# ("dental school at PSU" vs "online programs at PSU"). Removing it leaves the part that matters.
UNIVERSITY_NAME = re.compile(r"\b(at |in )?(pittsburg state university|pittsburg state|pitt state|psu)('s)?\b",
                             re.IGNORECASE)


def normalize_question(text):
    """'Does Pittsburg State University offer online programs?' -> 'Does offer online programs?'"""
    return re.sub(r'\s+', ' ', UNIVERSITY_NAME.sub(' ', text)).strip()


class Embedder:
    """Sentence embeddings with mean pooling and L2 normalization."""

    def __init__(self, path=RETRIEVER_PATH):
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModel.from_pretrained(path).eval()

    @torch.no_grad()
    def encode(self, texts):
        batch = self.tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors='pt')
        token_vectors = self.model(**batch).last_hidden_state           # (texts, tokens, 384)
        mask = batch['attention_mask'].unsqueeze(-1).float()            # ignore padding tokens
        sentence_vectors = (token_vectors * mask).sum(1) / mask.sum(1)  # average of real tokens
        return torch.nn.functional.normalize(sentence_vectors, dim=1)   # length 1, so dot = cosine


@dataclass
class Match:
    entry: Entry | None   # None when nothing is close enough
    score: float          # cosine similarity of the closest question (0..1 in practice)
    closest: Entry        # the closest entry even when it's below the threshold (for logs / eval)


class FaqRetriever:
    def __init__(self, embedder=None, entries=None, threshold=DEFAULT_THRESHOLD, normalize=True):
        self.embedder = embedder or Embedder()
        self.entries = entries or load_entries()
        self.threshold = threshold
        self.prepare = normalize_question if normalize else (lambda text: text)
        self.vectors = self.embedder.encode([self.prepare(e.question) for e in self.entries])

    def score_all(self, question):
        """Cosine similarity of the question to every dataset question."""
        return self.vectors @ self.embedder.encode([self.prepare(question)])[0]

    def search(self, question):
        scores = self.score_all(question)
        best = int(scores.argmax())
        score = float(scores[best])
        closest = self.entries[best]
        return Match(closest if score >= self.threshold else None, score, closest)
