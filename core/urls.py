from django.urls import path

from core.views import chatbot_response, generate_response, index

urlpatterns = [
    path('', index, name='index'),
    path('api/chat/', chatbot_response, name='chatbot_response'),         # retrieval (instant)
    path('api/generate/', generate_response, name='generate_response'),   # fine-tuned model (slow)
]
