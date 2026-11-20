"""Load the PSU Q&A dataset (data/programs.json) the same way for training, retrieval and the eval.

The raw file has 172 question/answer pairs written from pittstate.edu program pages. It repeats
some pairs, and 18 questions have two slightly different answers (e.g. "is an emphasis in the
Biology department" vs "offers an emphasis"). `load_entries()` merges those into one entry per
question and keeps the most detailed answer.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent.parent / 'data' / 'programs.json'

URL_PATTERN = re.compile(r'https?://\S+')
# "What is the Astrophysics program at ..." / "Tell me about the Art program at ..."
PROGRAM_PATTERN = re.compile(r'^(?:What is|Tell me about) the (.+?) program at Pittsburg State University')


@dataclass(frozen=True)
class Entry:
    question: str
    answer: str
    url: str | None        # the pittstate.edu link in the answer, if any
    program: str | None    # program name for the per-program questions, else None


def load_pairs(path=DATA_PATH):
    """All raw (question, answer) pairs, exact duplicates removed, original order kept."""
    seen, pairs = set(), []
    for item in json.loads(Path(path).read_text()):
        pair = (item['input'].strip(), item['response'].strip())
        if pair not in seen:
            seen.add(pair)
            pairs.append(pair)
    return pairs


def find_url(text):
    match = URL_PATTERN.search(text)
    return match.group(0).rstrip('.,)') if match else None


def load_entries(path=DATA_PATH):
    """One entry per distinct question, keeping the longest (most detailed) answer."""
    best = {}
    for question, answer in load_pairs(path):
        if question not in best or len(answer) > len(best[question]):
            best[question] = answer
    entries = []
    for question, answer in best.items():
        match = PROGRAM_PATTERN.match(question)
        entries.append(Entry(question, answer, find_url(answer), match.group(1) if match else None))
    return entries
