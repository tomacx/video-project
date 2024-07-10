from django.urls import path
from regional import views


urlpatterns = [
    path('regional',views.regional),
    path('video_feed',views.video_feed,name='video_feed'),
    path('video_feed2',views.video_feed2,name='video_feed2'),
    path('handle_points',views.handle_points,name='handle_points')
]