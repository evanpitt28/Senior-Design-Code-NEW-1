from django.contrib import admin
from django.urls import path, include, re_path
from . import views
from django.conf.urls.static import static 
from django.conf import settings 
from django.views.static import serve

app_name = 'posts'

urlpatterns = [
    path('', views.index, name='index'),
    path('data/', views.data_view, name='data-view'),
]
