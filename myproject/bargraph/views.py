from django.shortcuts import render

# Create your views here.

from django.http import JsonResponse
from django.db import models
from django.utils.timezone import now

def bargraph(request):
    return render(request, 'bargraph.html')

def data_view(request):
    # Your view logic here
    return render(request, 'data_template.html')

