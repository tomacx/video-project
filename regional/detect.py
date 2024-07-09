def detect(self, save_img=False, weights='weights/yolov5s.pt', source='data/images', img_size=640, conf_thres=0.25,
           iou_thres=0.45, device='', view_img=False, save_txt=False, save_conf=False,
           classes=None, agnostic_nms=False, augment=False, project='runs/detect', name='exp', exist_ok=False):
    imgsz = img_size

    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://'))

    # Directories
    save_dir = Path(increment_path(Path(project) / name, exist_ok=exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = save_img  # 保存视频
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        cv2.namedWindow('img')
        cv2.setMouseCallback('img', self.draw_ROI)

        # Process detections
        for i, det in enumerate(pred):  # detections per image 每一帧
            self.tableWidget.clearContents()
            newItem = QTableWidgetItem('总人数')
            self.tableWidget.setItem(0, 0, newItem)
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # display the resulting frame
                if (self.tempflag == True and self.drawing == False):  # 鼠标点击
                    try:
                        cv2.polylines(im0, self.pts[0], True, (0, 0, 255), thickness=2)
                    except:
                        pass
                    cv2.circle(im0, self.point1, 5, (0, 255, 0), 2)
                    for i in range(len(self.pointchoice) - 1):
                        cv2.line(im0, self.pointchoice[i], self.pointchoice[i + 1], (255, 0, 0), 2)
                if (self.tempflag == True and self.drawing == True):  # 鼠标右击
                    cv2.polylines(im0, self.pts[0], True, (0, 0, 255), thickness=2)
                if (self.tempflag == False and self.drawing == True):  # 鼠标中键
                    try:
                        cv2.polylines(im0, self.pts[0], True, (0, 0, 255), thickness=2)
                    except:
                        pass

                # Write results
                try:  # 计数
                    count_list = []
                    for l in range(len(self.pts[0])):
                        count_list.append(0)
                except:
                    pass
                total = 0  # 总人数
                man_list = []  # 计算人与人间距
                distence_list = []  # 人自身长度
                for *xyxy, conf, cls in reversed(det):  # 每一个识别到的人
                    if names[int(cls)] == 'person':
                        man_list.append(xyxy)
                        distence = ((int(xyxy[0]) - int(xyxy[2])) ** 2 + (int(xyxy[1]) - int(xyxy[3])) ** 2) ** 0.5
                        distence_list.append(distence)
                        total += 1
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        if save_img or view_img:  # Add bbox to image
                            try:
                                for row, pts in enumerate(self.pts[0]):
                                    c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                                    if if_inPoly(pts, c2):
                                        area = '区域' + str(row + 1)
                                        newItem = QTableWidgetItem(area)
                                        self.tableWidget.setItem(row + 1, 0, newItem)
                                        count_list[row] += 1
                                        newItem = QTableWidgetItem(str(count_list[row]))
                                        self.tableWidget.setItem(row + 1, 1, newItem)
                            except:
                                pass
                newItem = QTableWidgetItem(str(total))
                self.tableWidget.setItem(0, 1, newItem)
            # 计算距离
            if view_img:
                for index, dis_p in enumerate(man_list):
                    for other in man_list:
                        if dis_p != other:
                            c1, c2 = (int(dis_p[0]), int(dis_p[1])), (int(dis_p[2]), int(dis_p[3]))
                            distence2 = ((int(dis_p[2]) - int(other[2])) ** 2 + (
                                        int(dis_p[3]) - int(other[3])) ** 2) ** 0.5  # 两人间距
                            if distence_list[index] * 2 > distence2:
                                red = 255 * distence2 / (distence_list[index] * 2)
                                cv2.rectangle(im0, c1, c2, (0, 0, red), 1)
                                break
                            else:
                                cv2.rectangle(im0, c1, c2, (0, 255, 0), 1)

            # Print time (inference + NMS)
            # print(f'{s}Done. ({t2 - t1:.3f}s)')  # 打印人数等

            # Stream results
            if view_img:
                cv2.imshow('img', im0)
                cv2.waitKey(1)
                if cv2.waitKey(25) & 0xFF == 27:
                    vid_cap.release()
                    cv2.destroyAllWindows()
                    break

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fourcc = 'mp4v'  # output video codec
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")
    print(f'Done. ({time.time() - t0:.3f}s)')