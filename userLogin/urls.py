from django.urls import path
from userLogin import views
from userLogin.views import register,login
from .views import my_view
urlpatterns = [
    path('login', views.login),
    path('register', views.register),
    path('index',views.index),
    path('warning', my_view),
]
