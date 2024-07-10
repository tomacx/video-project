from django.contrib import admin
from django.urls import path
from faceRecog.views import capture_face, recognize_face, face, capture_face_start, recognize_face_start, show_face

urlpatterns = [
    path('capture_face_start', capture_face_start, name='capture_face_start'),#
    path('capture_face', capture_face, name='capture_face'),
    path('recognize_face_start', recognize_face_start, name='recognize_face_start'),
    path('recognize_face', recognize_face, name='recognize_face'),
    path('show_face',show_face, name='show_face')
]
