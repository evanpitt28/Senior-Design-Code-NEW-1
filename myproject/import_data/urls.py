# import_data/urls.py

from django.urls import path
from . import views  # Import views directly from the local directory

urlpatterns = [
    path('upload/', views.import_data_view, name='import_data_upload'),
]
