from django.contrib import admin
from django.urls import path
# from faceRecog.views import capture_face, recognize_face, face, capture_face_start, recognize_face_start
from faceRecog.views import recognize_face, recognize_face_start, capture_face_start, capture_face, take_photo, \
    show_face, show_face_search, capture_face_start_worker, recognize_face_start_worker

urlpatterns = [
    # path('faceRecog', face, name='face'),
    path('capture_face_start', capture_face_start, name='capture_face_start'),#
    path('capture_face_start_worker',capture_face_start_worker,name='capture_face_start_worker'),
    path('capture_face', capture_face, name='capture_face'),
    path('recognize_face_start', recognize_face_start, name='recognize_face_start'),
    path('recognize_face_start_worker',recognize_face_start_worker,name='recognize_face_start_worker'),
    path('recognize_face', recognize_face, name='recognize_face'),
    path('take_photo',take_photo,name='take_photo'),
    path('show_face',show_face,name='show_face'),
    path('show_face_search',show_face_search,name='show_face_search')
]
