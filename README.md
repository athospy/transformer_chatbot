# PSU Program Chatbot — a fine-tuned DialoGPT in Django

A small chatbot that answers questions about Pittsburg State University (PSU) academic programs.
It fine-tunes [microsoft/DialoGPT-small](https://huggingface.co/microsoft/DialoGPT-small)
(a 124M-parameter GPT-2 model) with Hugging Face Transformers and PyTorch, and serves it on CPU from a
Django app.

**Live demo:** https://chatbot.santiagomorel.dev/ (a 1-vCPU server, so an answer takes about 8–12 s)

This is a learning project from November 2024. Its value is the full loop: build a dataset, fine-tune,
save the model, serve it behind a web UI, and deploy it. The answers are **not** reliable. See
[Known limitations](#known-limitations) for why, and what I'd change.

## How it works

```
browser (templates/index.html + static/js/chat.js, no libraries)
   │  POST /api/chat/  {"message": "..."}
   ▼
Django view (core/views.py)
   │  tokenizer.encode(message + eos)  →  model.generate(...)  →  decode only the new tokens
   ▼
{"response": "...", "elapsed_ms": 9800}
```

- `core/views.py` loads the fine-tuned model and tokenizer once per process, at import time. Its
  `GENERATION_SETTINGS` are used for every answer and also listed on the page's "How it works" panel.
- Generation runs with 3 beams, at most 40 new tokens, no repeated 3-word sequences and a
  repetition penalty of 1.2. On the demo server's single CPU this takes about 8–12 s. The first version
  (5 beams, `max_length=100`) took 15–45 s.
- Input is cut to 300 characters. Errors return JSON: `405` for anything but POST, `400` for a body
  that isn't JSON or an empty message.
- The page (`static/js/chat.js`) adds every message with `textContent`, never as HTML, so neither user
  input nor model output can inject markup. It allows one request at a time, shows a seconds counter
  while the CPU works, gives up after 90 s, and offers "Try again" on errors. It suggests questions
  taken from the training topics, and its "How it works" panel explains the model and its limits.

## How it was fine-tuned

1. **Data** (`core/program_maker.py`): about 170 question/answer pairs written from PSU's program
   pages (`programs.json`: `{"input": ..., "response": ...}`). The script makes lightly reworded and
   shortened variants ("PSU" ↔ "Pittsburg State University", "programs" ↔ "courses", shorter phrasings),
   5 per pair, repeats the set 3 times, and writes a TSV.
   *The dataset files are not in this repo.*
2. **Training** (`core/training.py`, Hugging Face `Trainer` on CPU):
   - base model and tokenizer `microsoft/DialoGPT-small`, pad token = EOS
   - 80/20 train/validation split
   - AdamW (lr 2e-5, weight decay 0.01) with a cosine-annealing schedule
   - batch size 2 with gradient accumulation 8 (effective batch 16), up to 10 epochs
   - dropout 0.2, evaluation every 200 steps, early stopping after 3 evaluations without improvement,
     best checkpoint kept
   - the result is saved to `core/program_model/` and `core/program_tokenizer/`

## Run it locally

Needs Python 3.10+.

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

The fine-tuned weights (~500 MB) are not in the repo. To try the app without them, save the base
model into the folders the view loads from:

```bash
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
AutoTokenizer.from_pretrained('microsoft/DialoGPT-small').save_pretrained('core/program_tokenizer')
AutoModelForCausalLM.from_pretrained('microsoft/DialoGPT-small').save_pretrained('core/program_model')
"
```

For UI work, the tiny random model `sshleifer/tiny-gpt2` saved as `core/program_model` answers
instantly (with nonsense).

Or train your own: put a tab-separated `question<TAB>answer` file at `core/programs_simpler.tsv`,
`pip install -r requirements-train.txt`, then run `cd core && python training.py`.

Start the server:

```bash
DJANGO_DEBUG=1 python manage.py runserver
# open http://127.0.0.1:8000/
curl -s -X POST http://127.0.0.1:8000/api/chat/ \
  -H 'Content-Type: application/json' -d '{"message": "Does PSU offer online programs?"}'
```

### Configuration

Settings that differ per environment come from environment variables (see `.env.example`):

| Variable | Meaning |
|---|---|
| `DJANGO_SECRET_KEY` | Required unless `DJANGO_DEBUG=1`. The app refuses to start without it. |
| `DJANGO_ALLOWED_HOSTS` | Comma-separated host names (default `localhost,127.0.0.1`). |
| `DJANGO_DEBUG` | `1` for local development only. |

## Deployment (the live demo)

Ubuntu, nginx (TLS) → gunicorn over a Unix socket → Django, managed by systemd:

```
gunicorn --workers 1 --threads 4 --timeout 120 --bind unix:/run/transchat_gunicorn.sock chatbot.wsgi:application
```

- **One worker.** Each worker loads its own copy of the model and uses about 1 GB of RAM, and the
  server has 2 GB.
- **Four threads**, so the page and other requests still load while an answer is being generated.
- **`--timeout 120`**, plus `proxy_read_timeout 120s` in nginx. The first request after a restart
  also loads the model.

## Known limitations

- **The model doesn't really learn to answer the question.** In `training.py` the question and the
  answer are tokenized separately and lined up position by position (`labels = answer tokens`), so the
  model is trained to predict answer token *i* at question position *i*. A causal language model
  instead needs one sequence, `question <eos> answer <eos>`, with the loss computed only on the answer
  tokens (question positions set to `-100`). The result is a model that knows the vocabulary of PSU
  programs but often returns lists of program names instead of an answer.
- `DataCollatorWithLabelSmoothing` doesn't apply label smoothing, and the custom `compute_loss` uses
  plain cross-entropy, so the Trainer's `label_smoothing_factor` has no effect.
- ~170 source pairs is very little data for generation. Variants made by word swaps add little new
  information.
- No tests.

**What I'd do today:** fix the sequence format and loss masking above, and compare the result
against a retrieval baseline (embed the 170 answers and return the closest one). A retrieval baseline
is often the better tool for a fixed FAQ. An instruction-tuned model with the program list in its
context would be another baseline.

## Project layout

```
chatbot/            Django project (settings, urls, wsgi)
core/views.py       page + /api/chat/ endpoint
core/training.py    fine-tuning script
core/program_maker.py   data augmentation script
templates/index.html    chat UI
static/css/style.css
```
