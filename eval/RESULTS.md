# Eval results

76 test questions (41 more in dev, used only to choose the retrieval threshold). Built by `eval/build_eval_set.py`, scored by `eval/run_eval.py`.

## Retrieval (what the demo answers with)

| System | Threshold | Right answer (in data) | Paraphrase | Typo | General | Says "don't know" (not in data / off-topic) | Wrong answer given | ms/question |
|---|---|---|---|---|---|---|---|---|
| tfidf | 0.54 | 57.4% | 73.0% | 71.4% | 17.6% | 60.0% (55.6% / 66.7%) | 23.7% | 0.0 |
| embeddings | 0.56 | 82.0% | 91.9% | 71.4% | 64.7% | 73.3% (55.6% / 100.0%) | 13.2% | 2.4 |

## Generation (the fine-tuned DialoGPT-small models)

| Model | Exact official link | Names the program | Words shared with the reference (F1) | Invented links | Says "don't know" | ms/question |
|---|---|---|---|---|---|---|
| dialogpt-2024 | 0.0% | 18.2% | 0.074 | 0.0% | 0.0% | 964.1 |
| dialogpt-2026 | 45.5% | 47.7% | 0.674 | 56.6% | 0.0% | 625.0 |

Link and program columns use the per-program questions; F1 uses every in-data question.
Timings are on the machine that ran the eval, not the 1-CPU demo server.
