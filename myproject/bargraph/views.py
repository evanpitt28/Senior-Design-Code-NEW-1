from django.shortcuts import render

# Create your views here.

from django.http import JsonResponse
from .models import DataPoint

def data_view(request):
    data_points = DataPoint.objects.all()
    data = {
        "labels": [point.label for point in data_points],
        "values": [point.value for point in data_points],
    }
    return JsonResponse(data)