from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('api/predict', views.api_predict, name='api_predict'),
    path('batch', views.batch_predict, name='batch_predict'),
]