"""handing_app URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from handing_app import views

urlpatterns = [
    path("",views.index,name='index'),
    path("upload_files", views.upload_files, name='upload_files'),
    path('checkChunk/', views.checkChunk, name='checkChunk'),
    path('mergeChunks/', views.mergeChunks, name='mergeChunks'),
    path('upload/', views.upload, name='upload'),
    path('identify_model/', views.identify_model, name='identify_model'),
]
