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
        # Encode the new user input, add the eos_token and return a tensor in Pytorch
        inputs = tokenizer.encode(user_message + tokenizer.eos_token, return_tensors='pt').to(device)
        print(f"Tokenized input: {inputs}")

        # Generate a response using the model and the tokenized input
        # attention_mask is a binary tensor with 1 for real tokens and 0 for padding tokens
        attention_mask = torch.ones(inputs.shape, device=device)        
        
        outputs = model.generate(
            inputs,
            attention_mask=attention_mask,
            max_length=100,                         # max length of the output sequence is set to 100 because the maximum length that the model can handle is 1024
            num_return_sequences=1,                 # num_return_sequences is set to 1 because we only need one response
            no_repeat_ngram_size=2,                 # no_repeat_ngram_size is set to 2 to avoid repeating the same 2-grams in the output
            early_stopping=True,                    # early_stopping is set to True to stop the generation when the model outputs the first eos_token
            pad_token_id=tokenizer.eos_token_id,    # pad_token_id is set to the eos_token_id to pad the output if its length is less than max_length
            num_beams=5                             # num_beams is set to 5 to generate 5 beams and select the best one
        )
        print(f"Model outputs: {outputs}")

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Decoded response: {response}")

        return JsonResponse({'response':response})