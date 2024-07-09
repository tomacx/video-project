from django.urls import path
from regional import views


urlpatterns = [
    path('regional',views.regional),
    path('video_feed',views.video_feed,name='video_feed'),
    path('handle_points',views.handle_points,name='handle_points')
]