from django.urls import path
from userLogin import views
from userLogin.views import register,login
urlpatterns = [
    path('', login),
    path('register', register),
]
