"""Score every answering approach on eval/questions.json and write eval/results.json + eval/RESULTS.md.

Systems:
  tfidf       - retrieval by word overlap (TF-IDF cosine), a classic baseline
  embeddings  - retrieval by meaning (all-MiniLM-L6-v2), what the demo serves
  dialogpt-*  - a fine-tuned DialoGPT-small generating the answer (one entry per --model)

Retrieval systems get a similarity threshold, chosen on the dev split only: below it they say
"I don't know". Generators always answer, so they can't say "I don't know".

What counts as correct:
  questions about something in the data   right entry returned (retrieval); for generators, the
                                          exact official link (program questions) and how much of
                                          the reference answer it contains (token F1)
  questions about anything else           saying "I don't know"
  invented links                          a generated pittstate.edu link that isn't in the data

Run:  python eval/run_eval.py --model 2024=path/to/old_model --model 2026=core/program_model
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.faq_data import URL_PATTERN, load_entries  # noqa: E402
from core.generation import Generator  # noqa: E402
from core.retrieval import Embedder, FaqRetriever, normalize_question  # noqa: E402

QUESTIONS = ROOT / 'eval' / 'questions.json'
THRESHOLDS = [round(0.20 + 0.01 * i, 2) for i in range(71)]   # 0.20 .. 0.90


# ---------- Scoring helpers ----------

def tokens(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def token_f1(prediction, reference):
    """Share of words in common, as F1 (1.0 = same words, 0 = nothing in common)."""
    pred, ref = Counter(tokens(prediction)), Counter(tokens(reference))
    common = sum((pred & ref).values())
    if not common:
        return 0.0
    precision, recall = common / sum(pred.values()), common / sum(ref.values())
    return 2 * precision * recall / (precision + recall)


def pct(part, whole):
    return round(100 * part / whole, 1) if whole else None


# ---------- Retrieval systems ----------

class TfidfRetriever:
    """Word-overlap retrieval: TF-IDF vectors over the dataset questions, cosine similarity."""

    def __init__(self, entries, normalize=True):
        self.entries = entries
        self.prepare = normalize_question if normalize else (lambda text: text)
        docs = [Counter(tokens(self.prepare(e.question))) for e in entries]
        df = Counter(word for doc in docs for word in doc)
        self.idf = {word: math.log((1 + len(docs)) / (1 + n)) + 1 for word, n in df.items()}
        self.vectors = [self._vector(doc) for doc in docs]

    def _vector(self, counts):
        vec = {w: c * self.idf.get(w, 0.0) for w, c in counts.items()}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {w: v / norm for w, v in vec.items()}

    def score_all(self, question):
        q = self._vector(Counter(tokens(self.prepare(question))))
        return [sum(q.get(w, 0.0) * v for w, v in doc.items()) for doc in self.vectors]


def retrieval_rows(name, score_all, entries, items):
    """Closest entry + score per question; the threshold is applied later."""
    rows = []
    for item in items:
        started = time.perf_counter()
        scores = score_all(item['question'])
        best = max(range(len(scores)), key=scores.__getitem__)
        rows.append({'system': name, 'closest': entries[best].question, 'score': float(scores[best]),
                     'ms': (time.perf_counter() - started) * 1000, **item})
    return rows


def retrieval_correct(row, threshold):
    answered = row['score'] >= threshold
    if row['expected'] == 'none':
        return not answered
    return answered and row['closest'] == row['expected']


def pick_threshold(rows):
    """Threshold with the best dev accuracy; on ties, the middle of the best range."""
    dev = [r for r in rows if r['split'] == 'dev']
    accuracy = {t: sum(retrieval_correct(r, t) for r in dev) for t in THRESHOLDS}
    best = max(accuracy.values())
    best_range = [t for t in THRESHOLDS if accuracy[t] == best]
    return best_range[len(best_range) // 2], pct(best, len(dev))


def summarize_retrieval(rows, threshold):
    test = [r for r in rows if r['split'] == 'test']
    summary = {'threshold': threshold, 'ms_per_question': round(statistics.mean(r['ms'] for r in test), 1)}
    for kind in ('paraphrase', 'typo', 'general', 'not_in_data', 'off_topic'):
        subset = [r for r in test if r['kind'] == kind]
        summary[kind] = pct(sum(retrieval_correct(r, threshold) for r in subset), len(subset))
    in_scope = [r for r in test if r['expected'] != 'none']
    out_scope = [r for r in test if r['expected'] == 'none']
    summary['answered_correctly'] = pct(sum(retrieval_correct(r, threshold) for r in in_scope), len(in_scope))
    summary['says_dont_know'] = pct(sum(retrieval_correct(r, threshold) for r in out_scope), len(out_scope))
    summary['wrong_answer'] = pct(sum(r['score'] >= threshold and r['closest'] != r['expected'] for r in test),
                                  len(test))
    summary['overall'] = pct(sum(retrieval_correct(r, threshold) for r in test), len(test))
    summary['invented_links'] = 0.0
    return summary


# ---------- Generators ----------

def generate_rows(name, model_path, items, entries):
    generator = Generator(model_path)
    known_urls = {e.url for e in entries if e.url}
    by_question = {e.question: e for e in entries}
    rows = []
    for item in items:
        answer, ms = generator.answer(item['question'])
        expected = by_question.get(item['expected'])
        links = [u.rstrip('.,)') for u in URL_PATTERN.findall(answer)]
        rows.append({
            'system': name, 'answer': answer, 'ms': ms, **item,
            'right_link': bool(expected and expected.url and expected.url in links),
            'names_program': bool(expected and expected.program and expected.program.lower() in answer.lower()),
            'invented_link': any(u not in known_urls for u in links),
            'f1': token_f1(answer, expected.answer) if expected else None,
        })
    return rows


def summarize_generator(rows):
    test = [r for r in rows if r['split'] == 'test']
    program_rows = [r for r in test if r['kind'] in ('paraphrase', 'typo')]
    in_scope = [r for r in test if r['expected'] != 'none']
    general = [r for r in test if r['kind'] == 'general']
    return {
        'ms_per_question': round(statistics.mean(r['ms'] for r in test), 1),
        'right_link': pct(sum(r['right_link'] for r in program_rows), len(program_rows)),
        'names_program': pct(sum(r['names_program'] for r in program_rows), len(program_rows)),
        'token_f1_programs': round(statistics.mean(r['f1'] for r in program_rows), 3),
        'token_f1_general': round(statistics.mean(r['f1'] for r in general), 3),
        'token_f1_all': round(statistics.mean(r['f1'] for r in in_scope), 3),
        'invented_links': pct(sum(r['invented_link'] for r in test), len(test)),
        'says_dont_know': 0.0,
    }


# ---------- Report ----------

def write_markdown(results, path):
    lines = ['# Eval results', '',
             f"{results['counts']['test']} test questions ({results['counts']['dev']} more in dev, used only "
             'to choose the retrieval threshold). Built by `eval/build_eval_set.py`, scored by `eval/run_eval.py`.',
             '', '## Retrieval (what the demo answers with)', '',
             '| System | Threshold | Right answer (in data) | Paraphrase | Typo | General | Says "don\'t know" (not in data / off-topic) | Wrong answer given | ms/question |',
             '|---|---|---|---|---|---|---|---|---|']
    for name, s in results['retrieval'].items():
        lines.append(f"| {name} | {s['threshold']} | {s['answered_correctly']}% | {s['paraphrase']}% | {s['typo']}% | "
                     f"{s['general']}% | {s['says_dont_know']}% ({s['not_in_data']}% / {s['off_topic']}%) | "
                     f"{s['wrong_answer']}% | {s['ms_per_question']} |")
    lines += ['', '## Generation (the fine-tuned DialoGPT-small models)', '',
              '| Model | Exact official link | Names the program | Words shared with the reference (F1) | Invented links | Says "don\'t know" | ms/question |',
              '|---|---|---|---|---|---|---|']
    for name, s in results['generation'].items():
        lines.append(f"| {name} | {s['right_link']}% | {s['names_program']}% | {s['token_f1_all']} | "
                     f"{s['invented_links']}% | {s['says_dont_know']}% | {s['ms_per_question']} |")
    lines += ['', 'Link and program columns use the per-program questions; F1 uses every in-data question.',
              'Timings are on the machine that ran the eval, not the 1-CPU demo server.', '']
    path.write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', action='append', default=[], metavar='NAME=PATH',
                        help='a fine-tuned DialoGPT to score, e.g. 2024=old_model (repeatable)')
    parser.add_argument('--no-normalize', action='store_true',
                        help='keep "Pittsburg State University" etc. in questions (for comparison)')
    args = parser.parse_args()

    items = json.loads(QUESTIONS.read_text())
    entries = load_entries()
    results = {'counts': {s: sum(i['split'] == s for i in items) for s in ('dev', 'test')},
               'retrieval': {}, 'generation': {}}
    all_rows = []

    normalize = not args.no_normalize
    tfidf = TfidfRetriever(entries, normalize=normalize)
    embedder = FaqRetriever(Embedder(), entries, normalize=normalize)
    for name, score_all in (('tfidf', tfidf.score_all), ('embeddings', embedder.score_all)):
        rows = retrieval_rows(name, score_all, entries, items)
        threshold, dev_accuracy = pick_threshold(rows)
        results['retrieval'][name] = {**summarize_retrieval(rows, threshold), 'dev_accuracy': dev_accuracy}
        all_rows += rows
        print(f'{name}: threshold {threshold} (dev {dev_accuracy}%) -> test {results["retrieval"][name]["overall"]}%')

    for spec in args.model:
        name, path = spec.split('=', 1)
        rows = generate_rows(f'dialogpt-{name}', path, items, entries)
        results['generation'][f'dialogpt-{name}'] = summarize_generator(rows)
        all_rows += rows
        print(f'dialogpt-{name}:', results['generation'][f'dialogpt-{name}'])

    (ROOT / 'eval' / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
    (ROOT / 'eval' / 'answers.json').write_text(json.dumps(all_rows, indent=1, ensure_ascii=False) + '\n')
    write_markdown(results, ROOT / 'eval' / 'RESULTS.md')
    print('Wrote eval/results.json, eval/answers.json and eval/RESULTS.md')


if __name__ == '__main__':
    main()
