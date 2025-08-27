from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('api/predict', views.api_predict, name='api_predict'),
    path('batch', views.batch_predict, name='batch_predict'),
    path('sample.csv', views.sample_csv, name='sample_csv'),
    path('about', views.about, name='about'),
    path('insights', views.insights, name='insights'),
    path('ols', views.ols, name='ols'),  # NEW
]