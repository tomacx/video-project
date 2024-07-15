import threading
import time
import sendMessage
import cv2
from django.http import StreamingHttpResponse, JsonResponse, HttpResponse,HttpRequest
from django.shortcuts import render, redirect
import argparse
import csv
import os
import platform
import sys
from pathlib import Path
import torch

from yolov5.detect import parse_opt

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from ultralytics.utils.plotting import Annotator, colors, save_one_box

from yolov5.models.common import DetectMultiBackend
from yolov5.utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from yolov5.utils.general import (
    LOGGER,
    Profile,
    check_file,
    check_img_size,
    check_imshow,
    check_requirements,
    colorstr,
    cv2,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
    strip_optimizer,
    xyxy2xywh,
)
from yolov5.utils.torch_utils import select_device, smart_inference_mode

from userLogin.models import warn

thread_save = False
global number


def save(cam, name):
    print("in save")
    frame_width = int(cam.get(3))
    frame_height = int(cam.get(4))

    duration = 10
    t = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
    print(t)
    print(f'yolov5/warning/{name}-{t}.mp4')

    #设置编码器的格式为 H264-MPEG-4 AVC
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(f'yolov5/warning/{name}-{t}.mp4', fourcc, 10, (frame_width, frame_height))
    new_warn = warn(warningname=name, warningtime=t, savepath= f'yolov5/warning/{name}-{t}.mp4')
    new_warn.save()

    start_time = time.time()
    while True:
        ret, frame = cam.read()
        if ret:
            out.write(frame)
            if time.time() - start_time > duration:
                global thread_save
                thread_save = False
                print("finish save")
                break
        else:
            break
    out.release()


def save_video_thread(cam, name):
    save(cam, name)


def generate_frames(
        weights=ROOT / "yolov5s.pt",  # model path or triton URL
        source=ROOT / "data/images",  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / "data/coco128.yaml",  # fire.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_csv=False,  # save results in CSV format
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / "runs/detect",  # save results to project/name
        name="exp",  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
        person_num=[0],  # 视频中人数
):
    number = 0
    source = str(source)
    save_img = not nosave and not source.endswith(".txt")  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
    webcam = source.isnumeric() or source.endswith(".streams") or (is_url and not is_file)
    screenshot = source.lower().startswith("screen")
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(device=device), Profile(device=device), Profile(device=device))
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            if model.xml and im.shape[0] > 1:
                ims = torch.chunk(im, im.shape[0], 0)

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            if model.xml and im.shape[0] > 1:
                pred = None
                for image in ims:
                    if pred is None:
                        pred = model(image, augment=augment, visualize=visualize).unsqueeze(0)
                    else:
                        pred = torch.cat((pred, model(image, augment=augment, visualize=visualize).unsqueeze(0)), dim=0)
                pred = [pred, None]
            else:
                pred = model(im, augment=augment, visualize=visualize)
        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Define the path for the CSV file
        csv_path = save_dir / "predictions.csv"

        # Create or append to the CSV file
        def write_to_csv(image_name, prediction, confidence):
            """Writes prediction data for an image to a CSV file, appending if the file exists."""
            data = {"Image Name": image_name, "Prediction": prediction, "Confidence": confidence}
            with open(csv_path, mode="a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=data.keys())
                if not csv_path.is_file():
                    writer.writeheader()
                writer.writerow(data)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f"{i}: "
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, "frame", 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / "labels" / p.stem) + ("" if dataset.mode == "image" else f"_{frame}")  # im.txt
            s += "%gx%g " % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                person_num = 0
                knife_num = 0
                fire_num = 0
                fall_num = 0
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    if names[c] == 'person':
                        person_num += 1

                    label = names[c] if hide_conf else f"{names[c]}"
                    confidence = float(conf)
                    confidence_str = f"{confidence:.2f}"

                    global thread_save
                    if names[c] == 'knife' and confidence >= 0.40:
                        cam = cv2.VideoCapture(source)

                        if not thread_save:
                            thread_save = True
                            threading.Thread(target=save_video_thread, args=(cam, names[c])).start()
                        knife_num += 1
                        if number == 0:
                            number = number + 1
                            sendMessage.send_message()  # 发现危险物品报警

                    if names[c] == 'fire'  and confidence >= 0.40:
                        cam = cv2.VideoCapture(source)
                        if not thread_save:
                            thread_save = True
                            threading.Thread(target=save_video_thread, args=(cam, names[c])).start()
                        fire_num += 1
                        if number == 0:
                            number = number + 1
                            sendMessage.send_message()  # 发现火焰报警

                    if names[c] == 'falldown'  and confidence >= 0.40:
                        cam = cv2.VideoCapture(source)
                        if not thread_save:
                            thread_save = True
                            threading.Thread(target=save_video_thread, args=(cam, names[c])).start()
                        fall_num += 1
                        if number == 0:
                            number = number + 1
                            sendMessage.send_message()  # 发现有人摔倒报警

                    if save_csv:
                        write_to_csv(p.name, label, confidence_str)

                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f"{txt_path}.txt", "a") as f:
                            f.write(("%g " * len(line)).rstrip() % line + "\n")

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f"{names[c]} {conf:.2f}")
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / "crops" / names[c] / f"{p.stem}.jpg", BGR=True)

                if person_num > 0:
                    text = f'person: {person_num}'
                    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                    cv2.putText(im0, text, (640 - text_size[0] - 10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 255, 0), 2)
                if knife_num > 0:
                    text = f'warning! knife: {knife_num}'
                    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                    cv2.putText(im0, text, (640 - text_size[0] - 10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 255, 0), 2)
                if fire_num > 0:
                    text = f'warning! fire: {fire_num}'
                    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                    cv2.putText(im0, text, (640 - text_size[0] - 10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 255, 0), 2)
                if fall_num > 0:
                    text = f'warning! someone fall: {fall_num}'
                    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                    cv2.putText(im0, text, (640 - text_size[0] - 10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 255, 0), 2)

            # Stream results
            im0 = annotator.result()
            if view_img:
                retval, buffer = cv2.imencode('.jpg', im0)

                if not retval:
                    raise RuntimeError('Could not encode image')

                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def detect_object(request):
    opt = parse_opt("yolov5s.pt")

    return StreamingHttpResponse(generate_frames(**vars(opt)), content_type='multipart/x-mixed-replace; boundary=frame')


def detect_object_start(request):
    return render(request, 'detect_object.html')


def detect_act(request):
    opt = parse_opt("bestFalldown.pt")
    return StreamingHttpResponse(generate_frames(**vars(opt)), content_type='multipart/x-mixed-replace; boundary=frame')


def detect_act_start(request):
    return render(request, 'detect_act.html')


def detect_knife(request):
    opt = parse_opt("bestknife.pt")
    return StreamingHttpResponse(generate_frames(**vars(opt)), content_type='multipart/x-mixed-replace; boundary=frame')


def detect_knife_start(request):
    return render(request, 'detect_knife.html')


def detect_fire(request):
    opt = parse_opt("bestfire.pt")
    return StreamingHttpResponse(generate_frames(**vars(opt)), content_type='multipart/x-mixed-replace; boundary=frame')


def detect_fire_start(request):
    return render(request, 'detect_fire.html')


def warning(request):
    warning_list = warn.objects.all()
    return render(request, 'warning.html', {'warning_list': warning_list})


def delete_warning(request):
    warn_id = request.GET.get('id')  # 根据用户名删除用户
    warn_path = warn.objects.get(id=warn_id).savepath
    warn.objects.filter(id=warn_id).delete()

    path = "D:/summerProject2024/video/" + str(warn_path)
    os.remove(path)

    print(path)
    return redirect("/warning")  # 删除之后回到/userinfo/界面

def get_toggle_status(request):
    if request.method == 'GET':
        # 从数据库中获取当前的toggle状态
        warn_id = request.GET.get('id')
        warning = warn.objects.get(id=warn_id)
        return HttpResponse(warning.done)

def update_toggle_status(request):
    if request.method == 'POST':
        warn_id = request.POST.get('id')
        new_value = request.POST.get('value','0')

        if warn_id:
            warning = warn.objects.get(id=warn_id)
            warning.done = new_value
            warning.save()
            return HttpResponse('OK')
        else:
            return HttpResponse(status=400)


