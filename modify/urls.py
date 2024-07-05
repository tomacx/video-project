from django.urls import path
from modify import views


urlpatterns = [
    path('add',views.add),
    path('delete',views.delete),
    path('edit',views.edit),
    path('userinfo',views.userinfo),
    path('index',views.index)
]
