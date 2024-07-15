import torch
import numpy as np
import cv2
import threading
from django.shortcuts import render
from django.http import HttpResponse, StreamingHttpResponse

from tracking import sort
from tracking import utilities
from tracking import homography_tracker


video1_pth = 'tracking/data/test7.mp4'
video2_pth = 'tracking/data/test8.mp4'
# video1_pth = 'rtmp://116.62.245.164:1935/live'
# video2_pth = 'rtmp://116.62.245.164:1935/live/1'

homo_pth = 'tracking/data/test_matrix.npy'

video1 = None
video2 = None

detector = torch.hub.load("ultralytics/yolov5", "yolov5m")

cam4_H_cam1 = np.load(homo_pth)
cam1_H_cam4 = np.linalg.inv(cam4_H_cam1)

homographies = list()
homographies.append(np.eye(3))
homographies.append(cam1_H_cam4)

detector.agnostic = True

# Class 0 is Person
detector.classes = [0]
detector.conf = 0.30

trackers = [
    sort.Sort(
        max_age=30, min_hits=3, iou_threshold=0.3
    )
    for _ in range(2)
]
global_tracker = homography_tracker.MultiCameraTracker(homographies, iou_thres=0.20)

video = None

def index(request):

    return render(request, 'mc_mot.html')

def track_init(request):
    global video1
    global video2
    video1 = cv2.VideoCapture(video1_pth)
    assert video1.isOpened(), "Could not open video1 source"
    video2 = cv2.VideoCapture(video2_pth)
    assert video2.isOpened(), "Could not open video2 source"

    # 1000 was choosen arbitrarily
    feat_detector = cv2.SIFT_create(1000)

    _, frame1 = video1.read()
    _, frame2 = video2.read()

    kpts1, des1 = feat_detector.detectAndCompute(frame1, None)
    kpts2, des2 = feat_detector.detectAndCompute(frame2, None)

    bf = cv2.BFMatcher()

    # NOTE: k=2 means the euclidian distance between the two closest matches
    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    src_pts = np.float32([kpts1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kpts2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    cam4_H_cam1, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    np.save(homo_pth, cam4_H_cam1)

    src_pts = np.intp(src_pts).reshape(-1, 2)
    dst_pts = np.intp(dst_pts).reshape(-1, 2)

    img_with_matches = utilities.draw_matches(frame1, src_pts, frame2, dst_pts, good)

    assert cv2.imwrite("tracking/img_with_matches.png", img_with_matches)

    # NOTE: Second video 'cam4.mp4' is 17 frames behind the first video 'cam1.mp4'
    video2.set(cv2.CAP_PROP_POS_FRAMES, 45)

    return HttpResponse(200)

def read_frame(video_src, result):
    frame = video_src.read()[1]
    result[0] = frame


def gen():

    while True:

        # Get frames
        # frame1 = video1.read()[1]
        # frame2 = video2.read()[1]

        # Get frame by thread
        result1 = {}
        result2 = {}

        # create thread
        thread1 = threading.Thread(target=read_frame, args=(video1, result1))
        thread1.start()
        thread2 = threading.Thread(target=read_frame, args=(video2, result2))
        thread2.start()

        # 等待两个线程结束
        thread1.join()
        thread2.join()

        # 获取结果
        frame1 = result1[0]
        frame2 = result2[0]

        # NOTE: YoloV5 expects the images to be RGB instead of BGR
        frames = [frame1[:, :, ::-1], frame2[:, :, ::-1]]

        anno = detector(frames)

        dets, tracks = [], []


        for i in range(len(anno)):
            # Sort Tracker requires (x1, y1, x2, y2) bounding box shape
            det = anno.xyxy[i].cpu().numpy()
            det[:, :4] = np.intp(det[:, :4])
            dets.append(det)

            # Updating each tracker measures
            tracker = trackers[i].update(det[:, :4], det[:, -1])
            tracks.append(tracker)

        global_ids = global_tracker.update(tracks)

        for i in range(2):
            frames[i] = utilities.draw_tracks(
                frames[i][:, :, ::-1],
                tracks[i],
                global_ids[i],
                i,
                classes=detector.names,
            )

        vis = np.hstack(frames)
        resized_vis = cv2.resize(vis, (1280, 480))

        ret, jpeg = cv2.imencode('.jpg', resized_vis)
        frame_show = jpeg.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_show + b'\r\n\r\n')

def mc_mot_track(request):

    return StreamingHttpResponse(gen(), content_type='multipart/x-mixed-replace; boundary=frame')
