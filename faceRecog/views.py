import json
from urllib import response

import pickle
import dlib
import numpy as np
import cv2
import os
import pandas as pd
import shutil
import time
import logging
import tensorflow as tf
from django.http import HttpResponse, StreamingHttpResponse
from django.shortcuts import render

from faceRecog.features_extraction_to_csv import features_to_csv
from faceRecog.models import Face

global Face_Register_con

# 要读取人脸图像文件的路径 / Path of cropped faces
path_images_from_camera = "faceRecog/data/data_faces_from_camera/"
# Dlib 正向人脸检测器 / Use frontal face detector of Dlib
detector = dlib.get_frontal_face_detector()

# Dlib 人脸 landmark 特征点检测器 / Get face landmarks
predictor = dlib.shape_predictor('faceRecog/data/data_dlib/shape_predictor_68_face_landmarks.dat')

# Dlib Resnet 人脸识别模型, 提取 128D 的特征矢量 / Use Dlib resnet50 model to get 128D face descriptor
face_reco_model = dlib.face_recognition_model_v1("faceRecog/data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")

# def face(request):
#     return render(request, 'face.html')
#
def capture_face_start(request):
        return render(request, 'capture_face.html')

def capture_face_start_worker(request):
    return render(request, 'capture_face_worker.html')
def show_face(request):
    face_list = Face.objects.all()
    return render(request, "show_face.html", {"face_list": face_list})  ##将数据导入html模板中，进行数据渲染。


def show_face_search(request):
    if request.method == 'POST':
        user_name = request.POST.get('username')
        if len(user_name) == 0:
            face_list = Face.objects.all()
            return render(request, "show_face.html", {"face_list": face_list})  ##将数据导入html模板中，进行数据渲染。

        face_list = Face.objects.filter(username=user_name)
        return render(request, "show_face.html", {"face_list": face_list})  ##将数据导入html模板中，进行数据渲染。


class Face_Register:
    def __init__(self, username):
        self.path_photos_from_camera = "faceRecog/data/data_faces_from_camera/"
        self.current_face_dir = ''
        self.font = cv2.FONT_ITALIC
        self.input_name = username

        self.existing_faces_cnt = 0         # 已录入的人脸计数器 / cnt for counting saved faces
        self.ss_cnt = 0                     # 录入 personX 人脸时图片计数器 / cnt for screen shots
        self.current_frame_faces_cnt = 0    # 录入人脸计数器 / cnt for counting faces in current frame

        self.save_flag = 1                  # 之后用来控制是否保存图像的 flag / The flag to control if save
        self.press_s_flag = 0                    # 监控是否按s
        self.press_n_flag = 0               # 之后用来检查是否先按 'n' 再按 's' / The flag to check if press 'n' before 's'

        # FPS
        self.frame_time = 0
        self.frame_start_time = 0
        self.fps = 0
        self.fps_show = 0
        self.start_time = time.time()

    # 新建保存人脸图像文件和数据 CSV 文件夹 / Mkdir for saving photos and csv
    def pre_work_mkdir(self):
        # 新建文件夹 / Create folders to save face images and csv
        if os.path.isdir(self.path_photos_from_camera):
            pass
        else:
            os.mkdir(self.path_photos_from_camera)

    # 删除之前存的人脸数据文件夹 / Delete old face folders
    def pre_work_del_old_face_folders(self):
        # 删除之前存的人脸数据文件夹, 删除 "/data_faces_from_camera/person_x/"...
        folders_rd = os.listdir(self.path_photos_from_camera)
        for i in range(len(folders_rd)):
            shutil.rmtree(self.path_photos_from_camera+folders_rd[i])
        if os.path.isfile("faceRecog/data/features_all.csv"):
            os.remove("faceRecog/data/features_all.csv")

    # 如果有之前录入的人脸, 在之前 person_x 的序号按照 person_x+1 开始录入 / Start from person_x+1
    def check_existing_faces_cnt(self):
        if os.listdir("faceRecog/data/data_faces_from_camera/"):
            # 获取已录入的最后一个人脸序号 / Get the order of latest person
            person_list = os.listdir("faceRecog/data/data_faces_from_camera/")
            person_num_list = []
            for person in person_list:
                person_num_list.append(int(person.split('_')[-2]))
                print(str(person))
                print(str(person.split('_')[-1]))
            self.existing_faces_cnt = max(person_num_list)

        # 如果第一次存储或者没有之前录入的人脸, 按照 person_1 开始录入 / Start from person_1
        else:
            self.existing_faces_cnt = 0

    def mkdir_person(self):
        person_list = os.listdir("faceRecog/data/data_faces_from_camera/")
        flag = 0
        for person in person_list:
            if str(person.split('_')[-1]) == str(self.input_name):
                self.current_face_dir = self.path_photos_from_camera + str(person)
                flag = 1
                break
        if flag == 0:
            self.existing_faces_cnt += 1
            self.current_face_dir = self.path_photos_from_camera + "person_" + str(self.existing_faces_cnt) + '_' + str(self.input_name)
            os.makedirs(self.current_face_dir)
            print("新建的人脸文件夹 / Create folders:" + str(self.current_face_dir))
            new_face = Face(username=self.input_name, faceSave=self.current_face_dir)
            new_face.save()

        self.ss_cnt = 0  # 将人脸计数器清零
        self.press_n_flag = 1  # 已经按下 'n' / Pressed 'n' already


    def re_press_s(self):
        self.press_s_flag = 1

    # 更新 FPS / Update FPS of Video stream
    def update_fps(self):
        now = time.time()
        # 每秒刷新 fps / Refresh fps per second
        if str(self.start_time).split(".")[0] != str(now).split(".")[0]:
            self.fps_show = self.fps
        self.start_time = now
        self.frame_time = now - self.frame_start_time
        self.fps = 1.0 / self.frame_time
        self.frame_start_time = now

    # 生成的 cv2 window 上面添加说明文字 / PutText on cv2 window
    def draw_note(self, img_rd):
        # 添加说明 / Add some notes
        cv2.putText(img_rd, "Face Register", (20, 40), self.font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img_rd, "FPS:   " + str(self.fps_show.__round__(2)), (20, 100), self.font, 0.8, (0, 255, 0), 1,
                    cv2.LINE_AA)
        cv2.putText(img_rd, "Faces: " + str(self.current_frame_faces_cnt), (20, 140), self.font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)


    # 获取人脸 / Main process of face detection and saving
    def process(self, stream):
        # 1. 新建储存人脸图像文件目录 / Create folders to save photos
        self.pre_work_mkdir()

        # 2. 删除 "/data/data_faces_from_camera" 中已有人脸图像文件
        # / Uncomment if want to delete the saved faces and start from person_1
        # if os.path.isdir(self.path_photos_from_camera):
        #     self.pre_work_del_old_face_folders()

        # 3. 检查 "/data/data_faces_from_camera" 中已有人脸文件
        self.check_existing_faces_cnt()
        self.mkdir_person()

        while stream.isOpened():
            flag, img_rd = stream.read()        # Get camera video stream
            faces = detector(img_rd, 0)         # Use Dlib face detector

            # # 4. 按下 'n' 新建存储人脸的文件夹 / Press 'n' to create the folders for saving faces
            # if kk == ord('n'):
            #     self.mkdir_person()

            # 5. 检测到人脸 / Face detected
            if len(faces) != 0:
                # 矩形框 / Show the ROI of faces
                for k, d in enumerate(faces):
                    # 计算矩形框大小 / Compute the size of rectangle box
                    height = (d.bottom() - d.top())
                    width = (d.right() - d.left())
                    hh = int(height/2)
                    ww = int(width/2)

                    # 6. 判断人脸矩形框是否超出 480x640 / If the size of ROI > 480x640
                    if (d.right()+ww) > 640 or (d.bottom()+hh > 480) or (d.left()-ww < 0) or (d.top()-hh < 0):
                        cv2.putText(img_rd, "OUT OF RANGE", (20, 300), self.font, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
                        color_rectangle = (0, 0, 255)
                        save_flag = 0
                        if self.press_s_flag:
                            logging.warning("请调整位置 / Please adjust your position")
                    else:
                        color_rectangle = (255, 255, 255)
                        save_flag = 1

                    cv2.rectangle(img_rd,
                                  tuple([d.left() - ww, d.top() - hh]),
                                  tuple([d.right() + ww, d.bottom() + hh]),
                                  color_rectangle, 2)

                    # 7. 根据人脸大小生成空的图像 / Create blank image according to the size of face detected
                    img_blank = np.zeros((int(height*2), width*2, 3), np.uint8)

                    # 8. 按下 's' 保存摄像头中的人脸到本地 / Press 's' to save faces into local images
                    if save_flag:
                        if self.press_s_flag:
                            # 检查有没有先按'n'新建文件夹 / Check if you have pressed 'n'
                            if self.press_n_flag:
                                self.ss_cnt += 1
                                for ii in range(height * 2):
                                    for jj in range(width * 2):
                                        img_blank[ii][jj] = img_rd[d.top() - hh + ii][d.left() - ww + jj]
                                cv2.imwrite(self.current_face_dir + "/img_face_" + str(self.ss_cnt) + ".jpg", img_blank)
                                logging.info("%-40s %s/img_face_%s.jpg", "写入本地 / Save into：",
                                             str(self.current_face_dir), str(self.ss_cnt))
                                self.re_press_s()
                            else:
                                logging.warning("请先按 'N' 来建文件夹, 按 'S' / Please press 'N' and press 'S'")
                            self.press_s_flag = 0




            self.current_frame_faces_cnt = len(faces)

            # 9. 生成的窗口添加说明文字 / Add note on cv2 window
            self.draw_note(img_rd)

            # 11. Update FPS
            self.update_fps()

            ret, jpeg = cv2.imencode('.jpg', img_rd)
            frame = jpeg.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')



def capture_face(request):
    global Face_Register_con
    username = request.GET.get('username')
    logging.basicConfig(level=logging.INFO)
    Face_Register_con = Face_Register(username)
    rtmpUrl = 'rtmp://116.62.245.164:1935/live'
    # cap = cv2.VideoCapture(rtmpUrl)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Failed to open RTMP stream: " + rtmpUrl)

    Face_Register_con.process(cap)

    return StreamingHttpResponse(Face_Register_con.process(cap),
                                 content_type='multipart/x-mixed-replace; boundary=frame')

def take_photo(request):
    global Face_Register_con
    Face_Register_con.re_press_s()
    features_to_csv()

    return render(request,'index.html')

class Face_Recognizer:
    def __init__(self):
        self.font = cv2.FONT_ITALIC

        # FPS
        self.frame_time = 0
        self.frame_start_time = 0
        self.fps = 0
        self.fps_show = 0
        self.start_time = time.time()
        # cnt for frame
        self.frame_cnt = 0

        # 用来存放所有录入人脸特征的数组 / Save the features of faces in the database
        self.face_features_known_list = []
        # 存储录入人脸名字 / Save the name of faces in the database
        self.face_name_known_list = []

        # 用来存储上一帧和当前帧 ROI 的质心坐标 / List to save centroid positions of ROI in frame N-1 and N
        self.last_frame_face_centroid_list = []
        self.current_frame_face_centroid_list = []

        # 用来存储上一帧和当前帧检测出目标的名字 / List to save names of objects in frame N-1 and N
        self.last_frame_face_name_list = []
        self.current_frame_face_name_list = []

        # 上一帧和当前帧中人脸数的计数器 / cnt for faces in frame N-1 and N
        self.last_frame_face_cnt = 0
        self.current_frame_face_cnt = 0

        # 用来存放进行识别时候对比的欧氏距离 / Save the e-distance for faceX when recognizing
        self.current_frame_face_X_e_distance_list = []

        # 存储当前摄像头中捕获到的所有人脸的坐标名字 / Save the positions and names of current faces captured
        self.current_frame_face_position_list = []
        # 存储当前摄像头中捕获到的人脸特征 / Save the features of people in current frame
        self.current_frame_face_feature_list = []

        # e distance between centroid of ROI in last and current frame
        self.last_current_frame_centroid_e_distance = 0

        # 控制再识别的后续帧数 / Reclassify after 'reclassify_interval' frames
        # 如果识别出 "unknown" 的脸, 将在 reclassify_interval_cnt 计数到 reclassify_interval 后, 对于人脸进行重新识别
        self.reclassify_interval_cnt = 0
        self.reclassify_interval = 10

    # 从 "features_all.csv" 读取录入人脸特征 / Get known faces from "features_all.csv"
    def get_face_database(self):
        if os.path.exists("faceRecog/data/features_all.csv"):
            path_features_known_csv = "faceRecog/data/features_all.csv"
            csv_rd = pd.read_csv(path_features_known_csv, header=None)
            for i in range(csv_rd.shape[0]):
                features_someone_arr = []
                self.face_name_known_list.append(csv_rd.iloc[i][0])
                for j in range(1, 129):
                    if csv_rd.iloc[i][j] == '':
                        features_someone_arr.append('0')
                    else:
                        features_someone_arr.append(csv_rd.iloc[i][j])
                self.face_features_known_list.append(features_someone_arr)
            logging.info("Faces in Database： %d", len(self.face_features_known_list))
            return 1
        else:
            logging.warning("'features_all.csv' not found!")
            logging.warning("Please run 'get_faces_from_camera.py' "
                            "and 'features_extraction_to_csv.py' before 'face_reco_from_camera.py'")
            return 0

    def update_fps(self):
        now = time.time()
        # 每秒刷新 fps / Refresh fps per second
        if str(self.start_time).split(".")[0] != str(now).split(".")[0]:
            self.fps_show = self.fps
        self.start_time = now
        self.frame_time = now - self.frame_start_time
        self.fps = 1.0 / self.frame_time
        self.frame_start_time = now
    @staticmethod
    # 计算两个128D向量间的欧式距离 / Compute the e-distance between two 128D features
    def return_euclidean_distance(feature_1, feature_2):
        feature_1 = np.array(feature_1)
        feature_2 = np.array(feature_2)
        dist = np.sqrt(np.sum(np.square(feature_1 - feature_2)))
        return dist

    # 使用质心追踪来识别人脸 / Use centroid tracker to link face_x in current frame with person_x in last frame
    def centroid_tracker(self):
        for i in range(len(self.current_frame_face_centroid_list)):
            e_distance_current_frame_person_x_list = []
            # 对于当前帧中的人脸1, 和上一帧中的 人脸1/2/3/4/.. 进行欧氏距离计算 / For object 1 in current_frame, compute e-distance with object 1/2/3/4/... in last frame
            for j in range(len(self.last_frame_face_centroid_list)):
                self.last_current_frame_centroid_e_distance = self.return_euclidean_distance(
                    self.current_frame_face_centroid_list[i], self.last_frame_face_centroid_list[j])

                e_distance_current_frame_person_x_list.append(
                    self.last_current_frame_centroid_e_distance)

            last_frame_num = e_distance_current_frame_person_x_list.index(
                min(e_distance_current_frame_person_x_list))
            self.current_frame_face_name_list[i] = self.last_frame_face_name_list[last_frame_num]

    # 生成的 cv2 window 上面添加说明文字 / putText on cv2 window
    def draw_note(self, img_rd):
        # 添加说明 / Add some info on windows
        cv2.putText(img_rd, "Face Recognizer with OT", (20, 40), self.font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img_rd, "FPS:    " + str(self.fps.__round__(2)), (20, 130), self.font, 0.8, (0, 255, 0), 1,
                    cv2.LINE_AA)
        cv2.putText(img_rd, "Faces:  " + str(self.current_frame_face_cnt), (20, 160), self.font, 0.8, (0, 255, 0), 1,
                    cv2.LINE_AA)


    # 处理获取的视频流, 进行人脸识别 / Face detection and recognition wit OT from input video stream
    def process(self, stream):
        # 1. 读取存放所有人脸特征的 csv / Get faces known from "features.all.csv"
        if not self.get_face_database():
            print("Error: Cannot get faces from database")

        while stream.isOpened():
            self.frame_cnt += 1
            flag, img_rd = stream.read()
            kk = cv2.waitKey(1)

            # 加载活体检测模型
            liveness_model = tf.keras.models.load_model('faceRecog/liveness.model')
            le = pickle.loads(open('faceRecog/label_encoder.pickle', 'rb').read())


            # 2. 检测人脸 / Detect faces for frame X
            faces = detector(img_rd, 0)

            # 3. 更新人脸计数器 / Update cnt for faces in frames
            self.last_frame_face_cnt = self.current_frame_face_cnt
            self.current_frame_face_cnt = len(faces)

            # 4. 更新上一帧中的人脸列表 / Update the face name list in last frame
            self.last_frame_face_name_list = self.current_frame_face_name_list[:]

            # 5. 更新上一帧和当前帧的质心列表 / update frame centroid list
            self.last_frame_face_centroid_list = self.current_frame_face_centroid_list
            self.current_frame_face_centroid_list = []

            # 6.1 如果当前帧和上一帧人脸数没有变化 / if cnt not changes
            if (self.current_frame_face_cnt == self.last_frame_face_cnt) and (
                    self.reclassify_interval_cnt != self.reclassify_interval):
                logging.debug("scene 1: 当前帧和上一帧相比没有发生人脸数变化 / No face cnt changes in this frame!!!")

                self.current_frame_face_position_list = []

                if "unknown" in self.current_frame_face_name_list:
                    logging.debug("  有未知人脸, 开始进行 reclassify_interval_cnt 计数")
                    self.reclassify_interval_cnt += 1

                if self.current_frame_face_cnt != 0:
                    for k, d in enumerate(faces):
                        self.current_frame_face_position_list.append(tuple(
                            [faces[k].left(), int(faces[k].bottom() + (faces[k].bottom() - faces[k].top()) / 4)]))
                        self.current_frame_face_centroid_list.append(
                            [int(faces[k].left() + faces[k].right()) / 2,
                             int(faces[k].top() + faces[k].bottom()) / 2])

                        img_rd = cv2.rectangle(img_rd,
                                               tuple([d.left(), d.top()]),
                                               tuple([d.right(), d.bottom()]),
                                               (255, 255, 255), 2)

                # 如果当前帧中有多个人脸, 使用质心追踪 / Multi-faces in current frame, use centroid-tracker to track
                if self.current_frame_face_cnt != 1:
                    self.centroid_tracker()

                for i in range(self.current_frame_face_cnt):
                    # 6.2 Write names under ROI
                    img_rd = cv2.putText(img_rd, self.current_frame_face_name_list[i],
                                         self.current_frame_face_position_list[i], self.font, 0.8, (0, 255, 255), 1,
                                         cv2.LINE_AA)
                self.draw_note(img_rd)

            # 6.2 如果当前帧和上一帧人脸数发生变化 / If cnt of faces changes, 0->1 or 1->0 or ...
            else:
                logging.debug("scene 2: 当前帧和上一帧相比人脸数发生变化 / Faces cnt changes in this frame")
                self.current_frame_face_position_list = []
                self.current_frame_face_X_e_distance_list = []
                self.current_frame_face_feature_list = []
                self.reclassify_interval_cnt = 0

                # 6.2.1 人脸数减少 / Face cnt decreases: 1->0, 2->1, ...
                if self.current_frame_face_cnt == 0:
                    logging.debug("  scene 2.1 人脸消失, 当前帧中没有人脸 / No faces in this frame!!!")
                    # clear list of names and features
                    self.current_frame_face_name_list = []
                # 6.2.2 人脸数增加 / Face cnt increase: 0->1, 0->2, ..., 1->2, ...
                else:
                    logging.debug("  scene 2.2 出现人脸, 进行人脸识别 / Get faces in this frame and do face recognition")
                    self.current_frame_face_name_list = []
                    for i in range(len(faces)):
                        shape = predictor(img_rd, faces[i])
                        self.current_frame_face_feature_list.append(
                            face_reco_model.compute_face_descriptor(img_rd, shape))
                        self.current_frame_face_name_list.append("unknown")

                    # 6.2.2.1 遍历捕获到的图像中所有的人脸 / Traversal all the faces in the database
                    for k in range(len(faces)):
                        logging.debug("  For face %d in current frame:", k + 1)
                        self.current_frame_face_centroid_list.append(
                            [int(faces[k].left() + faces[k].right()) / 2,
                             int(faces[k].top() + faces[k].bottom()) / 2])

                        self.current_frame_face_X_e_distance_list = []

                        # 6.2.2.2 每个捕获人脸的名字坐标 / Positions of faces captured
                        self.current_frame_face_position_list.append(tuple(
                            [faces[k].left(), int(faces[k].bottom() + (faces[k].bottom() - faces[k].top()) / 4)]))

                        # 6.2.2.3 对于某张人脸, 遍历所有存储的人脸特征
                        # For every faces detected, compare the faces in the database
                        for i in range(len(self.face_features_known_list)):
                            # 如果 q 数据不为空
                            if str(self.face_features_known_list[i][0]) != '0.0':
                                e_distance_tmp = self.return_euclidean_distance(
                                    self.current_frame_face_feature_list[k],
                                    self.face_features_known_list[i])
                                logging.debug("      with person %d, the e-distance: %f", i + 1, e_distance_tmp)
                                self.current_frame_face_X_e_distance_list.append(e_distance_tmp)
                            else:
                                # 空数据 person_X
                                self.current_frame_face_X_e_distance_list.append(999999999)

                        # 6.2.2.4 寻找出最小的欧式距离匹配 / Find the one with minimum e distance
                        similar_person_num = self.current_frame_face_X_e_distance_list.index(
                            min(self.current_frame_face_X_e_distance_list))

                        if min(self.current_frame_face_X_e_distance_list) < 0.4:
                            self.current_frame_face_name_list[k] = self.face_name_known_list[similar_person_num]
                            logging.debug("  Face recognition result: %s",
                                          self.face_name_known_list[similar_person_num])
                        else:
                            logging.debug("  Face recognition result: Unknown person")
                    # 7. 生成的窗口添加说明文字 / Add note on cv2 window
                    self.draw_note(img_rd)

            # 活体检测
            for k, d in enumerate(faces):
                logging.debug("  For face %d in current frame:", k + 1)
                face = img_rd[d.top():d.bottom(), d.left():d.right()]
                # some error occur here if my face is out of frame and comeback in the frame
                try:
                    face = cv2.resize(face, (32, 32))  # our liveness model expect 32x32 input
                except:
                    break

                face = face.astype('float') / 255.0
                face = tf.keras.preprocessing.image.img_to_array(face)
                face = np.expand_dims(face, axis=0)
                preds = liveness_model.predict(face)[0]
                j = np.argmax(preds)
                label_name = le.classes_[j]  # get label of predicted class
                label = f'{label_name}: {preds[j]:.4f}'
                cv2.putText(img_rd, label, (d.left(), d.top() - 10),cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)

            self.update_fps()

            ret, jpeg = cv2.imencode('.jpg', img_rd)
            frame = jpeg.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def recognize_face_start(request):
    return render(request, 'recognize_face.html')
def recognize_face_start_worker(request):
    return render(request,'recognize_face_worker.html')

def recognize_face(request):
    logging.basicConfig(level=logging.INFO)
    Face_Recognizer_con = Face_Recognizer()
    rtmpUrl = 'rtmp://116.62.245.164:1935/live/2'
    # cap = cv2.VideoCapture(rtmpUrl)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Failed to open RTMP stream: " + rtmpUrl)

    Face_Recognizer_con.process(cap)

    return StreamingHttpResponse(Face_Recognizer_con.process(cap), content_type='multipart/x-mixed-replace; boundary=frame')