from django.urls import path
from .views import *

urlpatterns = [
    path('', home, name='home'),
    path('predict/', predict, name='predict'),
]