from django.urls import path
from userLogin import views
from userLogin.views import register,login
urlpatterns = [
    path('login', views.login),
    path('register', views.register),
    path('index',views.index)
]
