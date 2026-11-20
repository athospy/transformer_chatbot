# PSU Program Chatbot — retrieval vs. a fine-tuned DialoGPT, with an eval

A small chatbot that answers questions about Pittsburg State University (PSU) academic programs, built
three times:

1. **2024, v1:** ChatterBot "best match", which compares strings letter by letter.
2. **2024, v2:** [microsoft/DialoGPT-small](https://huggingface.co/microsoft/DialoGPT-small)
   (a 124M-parameter GPT-2 model) fine-tuned with Hugging Face Transformers and PyTorch. A data format
   bug meant it never learned to answer (see [What went wrong in 2024](#what-went-wrong-in-2024)).
3. **2026, v3 (this repo):** answers come from **retrieval** over sentence embeddings, which is instant,
   cites the official link, and says "I don't know" when it should. The **fine-tuned model was
   retrained the right way** and runs on demand next to it. An **eval** with 117 held-out questions
   measures both.

**Live demo:** https://chatbot.santiagomorel.dev/ (a 1-vCPU server: retrieval answers in milliseconds,
the model in about 10 s)

## Results

On 76 held-out test questions (`eval/RESULTS.md` has every column):

| System | Right answer* | Says "don't know" when it should | Invented links |
|---|---|---|---|
| **Retrieval by meaning (served)** | **82.0%** | **73.3%** | **0%** |
| Retrieval by word overlap (TF-IDF) | 57.4% | 60.0% | 0% |
| Fine-tuned DialoGPT, 2026 retrain | 45.5% | 0% | 56.6% |
| Fine-tuned DialoGPT, 2024 | 0.0% | 0% | 0.0% (it never wrote links) |

\* Retrieval: the right written answer. Models: the exact official link, on program questions.

- **Fixing the training format worked:** word overlap with the reference answers went from
  0.074 to 0.674 (token F1), and exact links from 0% to 45.5%.
- **But a small generative model invents things.** Asked "Can I study law at Pittsburg State?" it answers
  "…More details can be found here: pittstate.edu/law/programs/index.html", a page that doesn't exist. It
  invented a link in 8 of 9 questions about programs the data doesn't have, and it never says "I don't know".
- **So retrieval answers, and the model is shown next to it.** Retrieval is instant (2.4 ms
  per question vs about 625 ms for the model on an M-series Mac, about 10 s on the
  1-CPU server), only returns real links, and says "I don't know" below its threshold. Its weak spot:
  real-sounding programs the data doesn't cover (it said "I don't know" for 55.6% of them, but
  for 100.0% of off-topic questions).

## How it works

```
browser (templates/index.html + static/js/chat.js, no libraries)
   │
   ├─ POST /api/chat/      {"message": "..."}  → core/retrieval.py
   │     normalize ("Pittsburg State University" removed) → MiniLM embedding → closest question
   │     → {"found", "response", "source_url", "matched_question", "score", "elapsed_ms"}
   │
   └─ POST /api/generate/  {"message": "..."}  → core/generation.py   (button under each answer)
         "question<eos>" → fine-tuned DialoGPT-small → {"response", "elapsed_ms"}
```

- **Retrieval** (`core/retrieval.py`): the 82 distinct dataset questions are embedded once with
  [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) (mean pooling over
  plain `transformers`, so no extra package). A visitor's question is embedded the same way and
  compared by cosine similarity. Below the threshold (0.56, chosen on the dev split) the answer is
  "I don't know" plus suggestions, instead of the nearest wrong program.
- **Removing the university's name** before embedding matters: every dataset question ends in
  "at Pittsburg State University", which made unrelated questions look alike. On the eval it took
  embedding retrieval from 64.5% to 80.3% of test questions right.
- **Generation** (`core/generation.py`): the same settings for the website and the eval, 3 beams, at
  most 72 new tokens (9 in 10 answers fit, with their link), no repeated 3-word sequences, repetition
  penalty 1.2.
- **The page** (`static/js/chat.js`) inserts every message with `textContent`, never as HTML. Only
  links to `https://www.pittstate.edu/` that come from the dataset become clickable; links the model
  generates stay plain text. One request at a time, a seconds counter while the CPU works, a 90 s
  timeout, and "Try again" on errors. "How it works" on the page shows the eval table from
  `eval/results.json` and the training summary saved next to the model.
- Input is cut to 300 characters. Errors return JSON: `405` for anything but POST, `400` for a body
  that isn't a JSON object or an empty message.

## The data

`data/programs.json`: 172 question/answer pairs written in 2024 from PSU's public program pages
(pittstate.edu). After removing exact repeats there are 100 pairs and 82 distinct questions:

- 56 programs, one templated question each ("What is / Tell me about the X program at Pittsburg State
  University?"), each answer with the program's official link. **They only go from A to C**
  (Accelerated Pre-Med through Construction Technology). The 2024 scrape stopped there.
- 26 general questions: number of programs, online options, healthcare, engineering, certificates,
  admissions, financial aid, housing, library, athletics.

18 questions have two slightly different answers. `core/faq_data.py` merges them and keeps the more
detailed one.

## Fine-tuning (2026 retrain)

`core/training.py`, a plain PyTorch loop (MPS on a Mac, otherwise CPU or CUDA):

- each example is **one sequence**, `question <eos> answer <eos>`, and the **loss covers only the
  answer**: the question positions get the label `-100`, which cross-entropy ignores
- the 100 distinct pairs plus copies of each question with "PSU" and "Pitt State" (the answer stays
  the same); 10% of the distinct pairs held out for validation, with their copies kept out of training
- AdamW (lr 5e-5, weight decay 0.01), linear warmup over the first 10% of steps, batch size 8,
  gradient clipping at 1.0, early stopping after 3 epochs without a better validation loss, best epoch
  kept, seed 42
- saves the model with its tokenizer and `training_summary.json` (loss per epoch) to `core/program_model/`

The 2026 run: 268 training examples, 10 validation, best validation loss
0.6259 at epoch 4 of 7 (early stopping),
97 minutes on an Apple Silicon GPU (MPS).

**Tried and kept as is:** decoding without the repetition rules (`no_repeat_ngram_size`,
`repetition_penalty`) or with greedy search. On the dev split, links got a little more often right but
also more often invented. The differences were within noise on 41 questions, so the settings stayed.

## The eval

`eval/build_eval_set.py` writes `eval/questions.json`, 117 questions the models never trained on:

| Kind | What it tests | Right behavior |
|---|---|---|
| paraphrase (56) | each program in a new wording ("Can I study X at Pitt State?") | that program's answer |
| typo (11) | two letters swapped in the program name | that program's answer |
| general (26) | a hand-written rewording of each general question ("What clubs can I join?") | that answer |
| not in data (14) | programs the data doesn't cover: some PSU doesn't have (dentistry), some it does (Graphic Design) | "I don't know" |
| off-topic (10) | jokes, weather, and two near misses (car brakes, how computers work) | "I don't know" |

Within each kind, a third of the questions are **dev** (used only to choose the retrieval threshold and
to look at mistakes) and the rest are **test** (the numbers above). A unit test checks that no eval
question appears in the training data.

`eval/run_eval.py` scores:

- **Retrieval:** right entry returned, or "I don't know" for out-of-data questions.
- **Generation:** the exact official link on program questions, whether the program is named, word
  overlap with the reference answer (token F1), and invented links (a link that isn't in the data).

```bash
python eval/build_eval_set.py
python eval/run_eval.py --model 2024=path/to/2024_model --model 2026=core/program_model
```

Results go to `eval/RESULTS.md`, `eval/results.json` and every single answer to `eval/answers.json`.

**Tried and dropped:** also indexing each program's bare name ("Construction") as a second key, to
help near-identical names. No change on dev (90.2% either way), so the simpler version stayed.

## What went wrong in 2024

The 2024 `training.py` (in git history) tokenized the question and the answer separately and lined them
up position by position (`labels = answer tokens`), so the model learned to predict answer token *i* at
question position *i*. A causal language model needs one sequence, with the loss on the answer only.
The 2024 model knew the vocabulary of PSU programs, but returned lists of program names instead of
answers. Its custom data collator also didn't apply the label smoothing its name promised.

## Run it locally

Needs Python 3.10+.

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python scripts/download_retriever.py        # all-MiniLM-L6-v2 into core/retriever_model (~90 MB)
```

Then the generation model, either:

- **train it** (about 1.5 hours on an Apple Silicon Mac's GPU): `python core/training.py`
- **or download it** from Hugging Face (link added here once it is published)

For UI work, the tiny random model `sshleifer/tiny-gpt2` saved as `core/program_model` answers instantly
(with nonsense).

```bash
DJANGO_DEBUG=1 python manage.py runserver   # http://127.0.0.1:8000/
python -m unittest discover tests           # fast tests, no models needed
```

### Configuration

Settings that differ per environment come from environment variables (see `.env.example`):

| Variable | Meaning |
|---|---|
| `DJANGO_SECRET_KEY` | Required unless `DJANGO_DEBUG=1`. The app refuses to start without it. |
| `DJANGO_ALLOWED_HOSTS` | Comma-separated host names (default `localhost,127.0.0.1`). |
| `DJANGO_DEBUG` | `1` for local development only. |

## Deployment (the live demo)

Ubuntu, nginx (TLS) → gunicorn over a Unix socket → Django, managed by systemd, running as `www-data`:

```
gunicorn --workers 1 --threads 4 --timeout 120 --bind unix:/run/transchat_gunicorn.sock chatbot.wsgi:application
```

- **One worker.** The two models use about 1.1 GB of RAM per worker, and the server has 2 GB.
- **Four threads**, so the page and retrieval answers still load while the model is generating.
- **`--timeout 120`**, plus `proxy_read_timeout 120s` in nginx.
- `HF_HUB_OFFLINE=1`: the models are local files, so nothing is downloaded at runtime.

## Known limitations

- The data covers only 56 programs (A to C) and 26 general questions, written in 2024.
- Retrieval returns written answers only. It can't combine facts, and it misses synonyms the small
  embedding model doesn't know ("clubs" vs "student organizations").
- The fine-tuned model is a 124M-parameter model trained on 100 pairs. It can reproduce answers it
  has seen, but new wordings and unknown programs push it to blend answers, and it never says "I
  don't know".

## Project layout

```
data/programs.json        the 172 Q&A pairs
core/faq_data.py          loads them (one entry per question)
core/retrieval.py         embeddings + threshold  (/api/chat/)
core/generation.py        the fine-tuned model    (/api/generate/)
core/training.py          fine-tuning script (2026)
core/program_maker.py     2024 data augmentation script (kept for reference)
core/views.py, urls.py    the page and the two endpoints
eval/                     eval set builder, runner, results
tests/                    fast unit tests
templates/, static/       the chat page (plain HTML, CSS and JS)
scripts/download_retriever.py
```

## License

MIT (see `LICENSE`). The base models (DialoGPT-small, all-MiniLM-L6-v2) are MIT and Apache-2.0
licensed. The Q&A text comes from pittstate.edu's public program pages.
