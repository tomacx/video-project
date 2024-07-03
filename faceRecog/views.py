import json
from urllib import response

import cv2

from django.http import HttpResponse, StreamingHttpResponse
from django.shortcuts import render
from alg.face_taker import create_directory, get_face_id, save_name
from alg.face_train import train_face_alg
from alg.face_recognizer import recognizer_face_alg

def face(request):
    return render(request, 'face.html')

def capture_face_start(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        key = request.POST.get('string_key')
        context = {
            'username': username,
            'key': key,
        }
        return render(request, 'capture_face.html',context)

#捕捉照片
def capture_face(request):
    username = request.GET.get('username')
    key = request.GET.get('string_key')

    rtmpUrl = 'rtmp://116.62.245.164:1935/live/' + key
    print('Connecting ' + rtmpUrl)
    cam = cv2.VideoCapture(rtmpUrl)
    if not cam.isOpened():
        return HttpResponse('摄像头未打开')

    cam.set(3, 640)  # Set width
    cam.set(4, 480)  # Set height

    def gen_display(cam, username):
        directory = 'images'
        cascade_classifier_filename = 'haarcascade_frontalface_default.xml'
        names_json_filename = 'names.json'

        # Create 'images' directory if it doesn't exist
        create_directory(directory)

        # Load the pre-trained face cascade classifier
        faceCascade = cv2.CascadeClassifier(cascade_classifier_filename)

        # Initialize face capture variables
        count = 0
        face_id = get_face_id(directory)
        save_name(face_id, username, names_json_filename)
        print('\n[INFO] Initializing face capture. Look at the camera and wait...')

        while True:
            # Read a frame from the camera
            ret, img = cam.read()

            # Convert the frame to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Detect faces in the frame
            faces = faceCascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            # Process each detected face
            for (x, y, w, h) in faces:
                # Draw a rectangle around the detected face
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

                # Increment the count for naming the saved images
                count += 1

                # Save the captured image into the 'images' directory
                cv2.imwrite(f'./images/Users-{face_id}-{count}.jpg', gray[y:y + h, x:x + w])
                print("pic" + str(count))

                # Display the image with rectangles around faces
                ret, jpeg = cv2.imencode('.jpg', img)
                frame = jpeg.tobytes()

                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

            # Press Escape to end the program
            k = cv2.waitKey(100) & 0xff
            if k < 30:
                return

            elif count >= 30:#!!
                break
        cam.release()

    return StreamingHttpResponse(gen_display(cam, username),content_type='multipart/x-mixed-replace; boundary=frame')



# def train_face(request):
#     if request.method == 'POST':
#
#         return HttpResponse('Face train successful!')
#



def recognize_face_start(request):
    if request.method == 'POST':
        key = request.POST.get('string_key')
        context = {
            'key': key,
        }
        return render(request, 'recognize_face.html', context)


def recognize_face(request):
    train_face_alg()

    key = request.GET.get('string_key')
    rtmpUrl = 'rtmp://116.62.245.164:1935/live/' + key
    print('Connecting ' + rtmpUrl)
    cam = cv2.VideoCapture(rtmpUrl)
    if not cam.isOpened():
        print("Error: Failed to open RTMP stream: " + rtmpUrl)
        exit()

    cam.set(3, 640)  # Set width
    cam.set(4, 480)  # Set height

    def gen_display_recognize(cam):
        # Create LBPH Face Recognizer
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        # Load the trained model
        recognizer.read('trainer.yml')
        print(recognizer)
        # Path to the Haar cascade file for face detection
        face_cascade_Path = "haarcascade_frontalface_default.xml"

        # Create a face cascade classifier
        faceCascade = cv2.CascadeClassifier(face_cascade_Path)

        # Font for displaying text on the image
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Minimum width and height for the window size to be recognized as a face
        minW = 0.1 * cam.get(3)
        minH = 0.1 * cam.get(4)
        # Initialize user IDs and associated names
        id = 0
        # Don't forget to add names associated with user IDs
        names = ['None']
        with open('names.json', 'r') as fs:
            names = json.load(fs)
            names = list(names.values())

        # 读取图片
        while True:
            # Read a frame from the camera
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Detect faces in the frame
            faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=5,
                minSize=(int(minW), int(minH)),
            )

            for (x, y, w, h) in faces:

                # Draw a rectangle around the detected face
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Recognize the face using the trained model
                id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
                # Proba greater than 51
                if confidence > 51:
                    try:
                        # Recognized face
                        name = names[id]
                        confidence = "  {0}%".format(round(confidence))
                    except IndexError as e:
                        name = "Who are you?"
                        confidence = "N/A"
                else:
                    # Unknown face
                    name = "Who are you?"
                    confidence = "N/A"

                # Display the recognized name and confidence level on the image
                cv2.putText(img, name, (x + 5, y - 5), font, 1, (255, 255, 255), 2)
                cv2.putText(img, confidence, (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

                ret, jpeg = cv2.imencode('.jpg', img)
                frame = jpeg.tobytes()

                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    return StreamingHttpResponse(gen_display_recognize(cam), content_type='multipart/x-mixed-replace; boundary=frame')

