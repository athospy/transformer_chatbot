"""The chat page and its two JSON endpoints.

  POST /api/chat/      retrieval: the closest written answer from the dataset, in milliseconds,
                       or "I don't know" when nothing is close enough (core/retrieval.py)
  POST /api/generate/  the fine-tuned DialoGPT-small's own answer, ~5-15 s on CPU (core/generation.py)

Both models load once per process, when this module is imported.
"""
import json
import os
import time
from pathlib import Path

from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt

from core.faq_data import load_entries
from core.generation import GENERATION_SETTINGS, Generator
from core.retrieval import FaqRetriever

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'program_model')
TOKENIZER_PATH = os.path.join(BASE_DIR, 'program_tokenizer')
EVAL_RESULTS = Path(BASE_DIR).parent / 'eval' / 'results.json'

retriever = FaqRetriever()
generator = Generator(MODEL_PATH, TOKENIZER_PATH)

# Longest question we accept. Longer input means slower generation on a small CPU server.
MAX_MESSAGE_CHARS = 300

# Clickable starters on the page: questions the data covers, in new wordings. Each one was
# checked to retrieve the right answer (a starter that says "I don't know" would be a bad first look).
SUGGESTED_QUESTIONS = [
    'Can I study Actuarial Science at PSU?',
    'Can I take PSU courses online?',
    'Which engineering degrees does PSU have?',
    'Is there a Ceramics certificate?',
    'What housing options does PSU have?',
]

DONT_KNOW = ("I don't have that in my data. I know 56 PSU programs (from A to C) and general "
             "questions about the university. Try one of the suggested questions.")

ENTRIES = load_entries()
DATA_SUMMARY = {
    'entries': len(ENTRIES),
    'programs': sum(1 for e in ENTRIES if e.program),
}


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


# Rows for the eval table on the "How it works" panel, from eval/results.json (run_eval.py)
SYSTEM_LABELS = {
    'embeddings': 'Retrieval by meaning (served)',
    'tfidf': 'Retrieval by word overlap',
    'dialogpt-2026': 'Fine-tuned model, 2026 retrain',
    'dialogpt-2024': 'Fine-tuned model, 2024',
}


def eval_table():
    results = read_json(EVAL_RESULTS)
    if not results:
        return None
    rows = []
    for name, s in results['retrieval'].items():
        rows.append({'label': SYSTEM_LABELS.get(name, name), 'right': s['answered_correctly'],
                     'dont_know': s['says_dont_know'], 'invented': s['invented_links']})
    for name, s in results['generation'].items():
        rows.append({'label': SYSTEM_LABELS.get(name, name), 'right': s['right_link'],
                     'dont_know': s['says_dont_know'], 'invented': s['invented_links']})
    order = list(SYSTEM_LABELS.values())
    rows.sort(key=lambda r: order.index(r['label']) if r['label'] in order else len(order))
    return {'rows': rows, 'test_questions': results['counts']['test'], 'dev_questions': results['counts']['dev']}


def training_info():
    """core/program_model/training_summary.json (written by core/training.py), plus the kept epoch."""
    summary = read_json(os.path.join(MODEL_PATH, 'training_summary.json'))
    if summary and summary.get('history'):
        summary['best_epoch'] = min(summary['history'], key=lambda h: h['val_loss'])['epoch']
    return summary


def index(request):
    return render(request, 'index.html', {
        'suggested_questions': SUGGESTED_QUESTIONS,
        'generation_settings': GENERATION_SETTINGS,
        'max_message_chars': MAX_MESSAGE_CHARS,
        'threshold': retriever.threshold,
        'data_summary': DATA_SUMMARY,
        'eval': eval_table(),
        'training': training_info(),
    })


def read_message(request):
    """The visitor's message, or a JsonResponse error to return instead."""
    if request.method != 'POST':
        return None, JsonResponse({'error': 'POST only'}, status=405)
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return None, JsonResponse({'error': 'Body must be JSON'}, status=400)
    if not isinstance(data, dict):
        return None, JsonResponse({'error': 'Body must be a JSON object'}, status=400)
    message = str(data.get('message') or '').strip()[:MAX_MESSAGE_CHARS]
    if not message:
        return None, JsonResponse({'error': 'Empty message'}, status=400)
    return message, None


# CSRF protection is off on purpose: these public endpoints have no login, cookies or state
# to protect, they only return text.
@csrf_exempt
def chatbot_response(request):
    message, error = read_message(request)
    if error:
        return error

    started = time.perf_counter()
    match = retriever.search(message)
    elapsed_ms = round((time.perf_counter() - started) * 1000)

    return JsonResponse({
        'found': match.entry is not None,
        'response': match.entry.answer if match.entry else DONT_KNOW,
        'source_url': match.entry.url if match.entry else None,
        'matched_question': match.closest.question,     # shown on the page, even below the threshold
        'score': round(match.score, 3),
        'threshold': retriever.threshold,
        'elapsed_ms': elapsed_ms,
    })


@csrf_exempt
def generate_response(request):
    message, error = read_message(request)
    if error:
        return error
    answer, elapsed_ms = generator.answer(message)
    return JsonResponse({'response': answer or '(the model returned nothing)', 'elapsed_ms': elapsed_ms})
