"""Fine-tune microsoft/DialoGPT-small on the PSU Q&A pairs (data/programs.json).

Each example is ONE sequence, the way DialoGPT reads a conversation:

    question <eos> answer <eos>

and the loss is computed only on the answer tokens: the question positions get the label -100,
which PyTorch's cross-entropy ignores. So the model learns "after this question, say this answer".

The 2024 version of this script (see git history) tokenized the question and the answer
separately and lined them up position by position, so the model learned to predict answer word i
at question word i. That's why the 2024 model returned lists of program names instead of answers.

Run from the repo root (uses the Mac's GPU through MPS when available, else CPU):
    python core/training.py                         # -> core/program_model, core/program_tokenizer
    python core/training.py --epochs 3 --out /tmp/m # quick try
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from core.faq_data import load_pairs  # noqa: E402

BASE_MODEL = 'microsoft/DialoGPT-small'


def with_short_names(pairs):
    """Add a copy of each question with the short names people use ("PSU", "Pitt State").

    The answers stay the same. The eval's questions are written separately (eval/build_eval_set.py),
    so this doesn't leak them into training.
    """
    extra = []
    for question, answer in pairs:
        for short in ('PSU', 'Pitt State'):
            variant = question.replace('Pittsburg State University', short)
            if variant != question:
                extra.append((variant, answer))
    return pairs + extra


def encode(tokenizer, question, answer):
    """Token ids for 'question<eos>answer<eos>' and labels that only score the answer."""
    question_ids = tokenizer.encode(question + tokenizer.eos_token)
    answer_ids = tokenizer.encode(answer + tokenizer.eos_token)
    return question_ids + answer_ids, [-100] * len(question_ids) + answer_ids


def batches(examples, batch_size, pad_id, shuffle):
    """Pad each batch to its longest example. Padding is masked out of attention and of the loss."""
    order = list(range(len(examples)))
    if shuffle:
        random.shuffle(order)
    for start in range(0, len(order), batch_size):
        chunk = [examples[i] for i in order[start:start + batch_size]]
        width = max(len(ids) for ids, _ in chunk)
        input_ids = [ids + [pad_id] * (width - len(ids)) for ids, _ in chunk]
        labels = [lab + [-100] * (width - len(lab)) for _, lab in chunk]
        mask = [[1] * len(ids) + [0] * (width - len(ids)) for ids, _ in chunk]
        yield torch.tensor(input_ids), torch.tensor(mask), torch.tensor(labels)


def mean_loss(model, examples, batch_size, pad_id, device):
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for input_ids, mask, labels in batches(examples, batch_size, pad_id, shuffle=False):
            out = model(input_ids=input_ids.to(device), attention_mask=mask.to(device), labels=labels.to(device))
            total += out.loss.item() * len(input_ids)
            count += len(input_ids)
    return total / count


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--epochs', type=int, default=20, help='upper limit; early stopping usually ends sooner')
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--patience', type=int, default=3, help='stop after this many epochs without a better validation loss')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out', default=str(ROOT / 'core' / 'program_model'))
    parser.add_argument('--tokenizer-out', default=str(ROOT / 'core' / 'program_tokenizer'))
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL).to(device)

    # Hold out 10% of the distinct pairs to watch for overfitting. Their short-name copies stay
    # out of training too, so validation never sees a near-copy of a training question.
    pairs = load_pairs()
    random.shuffle(pairs)
    n_val = max(1, len(pairs) // 10)
    val_pairs, train_pairs = pairs[:n_val], with_short_names(pairs[n_val:])
    train = [encode(tokenizer, q, a) for q, a in train_pairs]
    val = [encode(tokenizer, q, a) for q, a in val_pairs]
    print(f'{len(train)} training examples, {len(val)} validation, device {device}')

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = args.epochs * ((len(train) + args.batch_size - 1) // args.batch_size)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=steps // 10, num_training_steps=steps)
    pad_id = tokenizer.eos_token_id    # GPT-2 has no pad token; padding is masked anyway

    best_val, best_state, since_best, history = float('inf'), None, 0, []
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss, seen = 0.0, 0
        for input_ids, mask, labels in batches(train, args.batch_size, pad_id, shuffle=True):
            # The model shifts the labels by one inside, so position t learns to predict token t+1
            out = model(input_ids=input_ids.to(device), attention_mask=mask.to(device), labels=labels.to(device))
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            train_loss += out.loss.item() * len(input_ids)
            seen += len(input_ids)
        val_loss = mean_loss(model, val, args.batch_size, pad_id, device)
        history.append({'epoch': epoch, 'train_loss': round(train_loss / seen, 4), 'val_loss': round(val_loss, 4)})
        print(f"epoch {epoch:2}  train loss {history[-1]['train_loss']:.3f}  validation loss {val_loss:.3f}")
        if val_loss < best_val:
            best_val, since_best = val_loss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            since_best += 1
            if since_best >= args.patience:
                print(f'No improvement for {args.patience} epochs, stopping.')
                break

    model.load_state_dict(best_state)    # keep the epoch with the best validation loss
    model.to('cpu').save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)  # the model folder is self-contained (handy for Hugging Face)
    tokenizer.save_pretrained(args.tokenizer_out)
    summary = {
        'base_model': BASE_MODEL, 'seed': args.seed, 'lr': args.lr, 'batch_size': args.batch_size,
        'train_examples': len(train), 'val_examples': len(val), 'device': device,
        'best_val_loss': round(best_val, 4), 'minutes': round((time.time() - started) / 60, 1), 'history': history,
    }
    Path(args.out, 'training_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(f"Saved to {args.out} (best validation loss {best_val:.3f}, {summary['minutes']} min)")


if __name__ == '__main__':
    main()
