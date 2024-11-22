from django.shortcuts import render
from transformers import AutoModelForCausalLM , AutoTokenizer
from django.views.decorators.csrf import csrf_exempt
import torch
import json
from django.http import JsonResponse
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

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



def index(request):
    return render(request,'index.html')


@csrf_exempt
def chatbot_response(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        user_message = data.get('message')

       # Tokenize user input and generate output response
        inputs = tokenizer.encode(user_message + tokenizer.eos_token, return_tensors='pt').to(device)
        print(f"Tokenized input: {inputs}")

        attention_mask = torch.ones(inputs.shape, device=device)
        outputs = model.generate(
            inputs,
            attention_mask=attention_mask,
            max_length=100,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            early_stopping=True,
            pad_token_id=tokenizer.eos_token_id,
            num_beams=5  # Set num_beams to a value greater than 1
        )
        print(f"Model outputs: {outputs}")

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Decoded response: {response}")

        return JsonResponse({'response':response})