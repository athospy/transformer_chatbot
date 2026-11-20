"""Download the retrieval embedding model into core/retriever_model/ (~90 MB, done once).

Run:  python scripts/download_retriever.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from transformers import AutoModel, AutoTokenizer  # noqa: E402

from core.retrieval import RETRIEVER_HUB_ID, RETRIEVER_PATH  # noqa: E402

AutoTokenizer.from_pretrained(RETRIEVER_HUB_ID).save_pretrained(RETRIEVER_PATH)
AutoModel.from_pretrained(RETRIEVER_HUB_ID).save_pretrained(RETRIEVER_PATH)
print(f'Saved {RETRIEVER_HUB_ID} to {RETRIEVER_PATH}')
