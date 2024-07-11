from django.urls import path
from modify import views


urlpatterns = [
    path('add',views.add),
    path('delete',views.delete),
    path('edit',views.edit),
    path('userinfo',views.userinfo),
    path('userinfo_worker', views.userinfo_worker, name='userinfo_worker'),
    path('index',views.index),
    path('index_worker',views.index_worker)
]
