from django.urls import path
from tracking.views import mc_mot_track, track_init
urlpatterns = [
    path('mc_mot_track', mc_mot_track, name='mc_mot_track'),
    path('track_init', track_init, name='track_init')
]