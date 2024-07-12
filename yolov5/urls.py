from django.urls import path
from yolov5 import views

urlpatterns = [
    path('detect_object', views.detect_object, name='detect_object'),
    path('detect_object_start', views.detect_object_start, name='detect_object_start'),
    path('detect_act', views.detect_act, name='detect_act'),
    path('detect_act_start', views.detect_act_start, name='detect_act_start'),
    path('detect_knife', views.detect_knife, name='detect_knife'),
    path('detect_knife_start', views.detect_knife_start,name='detect_knife_start'),
    path('detect_fire', views.detect_fire, name='detect_fire'),
    path('detect_fire_start', views.detect_fire_start, name='detect_fire_start'),
    path('warning',views.warning,name='warning'),
    path('delete_warning',views.delete_warning,name='delete_warning'),
    path('get_toggle_status',views.get_toggle_status,name='get_toggle_status'),
    path('update_toggle_status',views.update_toggle_status,name='get_toggle_status')
]