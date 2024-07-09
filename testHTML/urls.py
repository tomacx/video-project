from django.urls import path
from testHTML import views

urlpatterns = [
    path('tables', views.tables)
]