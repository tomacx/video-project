import os
import subprocess

from django.shortcuts import render
from django.http import StreamingHttpResponse, HttpResponse, JsonResponse, HttpResponseBadRequest
from regional import detect0406
import sys
import cv2
import json



# Create your views here.
def regional(request):
    if request.method == 'POST':
            app = detect0406.QApplication(sys.argv)
            win = detect0406.Main()
            win.show()
            sys.exit(app.exec_())

    return render(request, 'regional.html')

def video_feed(request):
    cap = cv2.VideoCapture("rtmp://116.62.245.164:1935/live")

    def generate_frames():
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            _, jpeg = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

    return StreamingHttpResponse(generate_frames(), content_type='multipart/x-mixed-replace; boundary=frame')
