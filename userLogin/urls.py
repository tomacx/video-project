from django.urls import path
from userLogin import views

urlpatterns = [
    path('', views.login, name='login'),
]
