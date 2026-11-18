from django.shortcuts import render
from transformers import AutoModelForCausalLM , AutoTokenizer
from django.views.decorators.csrf import csrf_exempt
import torch
import json
from django.http import JsonResponse
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import time

# tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
# model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")

# Get the absolute paths to the model and tokenizer directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TOKENIZER_PATH = os.path.join(BASE_DIR, "program_tokenizer")
MODEL_PATH = os.path.join(BASE_DIR, "program_model")

# print(f"BASE_DIR: {BASE_DIR}, TOKENIZER_PATH: {TOKENIZER_PATH}, MODEL_PATH: {MODEL_PATH}")

# Load the saved model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

# Setup device for model (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Generation settings tuned on the 1-CPU demo server (2026-11-17): 3 beams and at most
# 40 new tokens answer in about 8-12 s; the old 5 beams / max_length=100 took 15-45 s and
# hit gunicorn's timeout. The page shows them in its "How it works" panel.
GENERATION_SETTINGS = {
    'max_new_tokens': 40,          # cap the answer length (not counting the question)
    'num_beams': 3,                # keep the 3 best partial answers at each step
    'early_stopping': True,        # stop when every beam has finished its answer
    'no_repeat_ngram_size': 3,     # never repeat the same 3 words in a row
    'repetition_penalty': 1.2,     # discourage repeating words the model already used
}

# Longest question we accept. Longer input means slower generation on a small CPU server.
MAX_MESSAGE_CHARS = 300

# Clickable starters on the page. They come from the topics in the training data
# (core/program_maker.py), so visitors ask what the model has actually seen.
SUGGESTED_QUESTIONS = [
    'What engineering programs does PSU offer?',
    'Does PSU offer online programs?',
    'Tell me about healthcare programs.',
    'Are certificate programs available at PSU?',
    'What business programs does PSU have?',
]

# Shown when the model returns nothing but punctuation or whitespace.
EMPTY_ANSWER = "I don't have an answer for that. Try one of the suggested questions."


def index(request):
    return render(request, 'index.html', {
        'suggested_questions': SUGGESTED_QUESTIONS,
        'generation_settings': GENERATION_SETTINGS,
        'max_message_chars': MAX_MESSAGE_CHARS,
    })


# CSRF protection is off on purpose: this public endpoint has no login, cookies or state
# to protect, it only returns generated text.
@csrf_exempt
def chatbot_response(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'POST only'}, status=405)

    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Body must be JSON'}, status=400)

    if not isinstance(data, dict):
        return JsonResponse({'error': 'Body must be a JSON object'}, status=400)

    user_message = str(data.get('message') or '').strip()[:MAX_MESSAGE_CHARS]
    if not user_message:
        return JsonResponse({'error': 'Empty message'}, status=400)

    # Encode the user input, add the eos_token (DialoGPT's end-of-turn marker) and
    # return a PyTorch tensor
    inputs = tokenizer.encode(user_message + tokenizer.eos_token, return_tensors='pt').to(device)

    # attention_mask is a binary tensor with 1 for real tokens and 0 for padding tokens
    attention_mask = torch.ones(inputs.shape, device=device)

    started = time.perf_counter()
    with torch.no_grad():  # inference only, no gradients needed
        outputs = model.generate(
            inputs,
            attention_mask=attention_mask,
            pad_token_id=tokenizer.eos_token_id,    # GPT-2 has no pad token, so pad with eos
            **GENERATION_SETTINGS,
        )
    elapsed_ms = round((time.perf_counter() - started) * 1000)

    # Decode only the new tokens, so the answer doesn't start with the user's own question
    response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
    response = response.strip().lstrip('.').strip() or EMPTY_ANSWER

    return JsonResponse({'response': response, 'elapsed_ms': elapsed_ms})
