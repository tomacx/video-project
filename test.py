import cv2

def main():
    cap = cv2.VideoCapture("rtmp://116.62.245.164/live")  # 高配版
    if not cap.isOpened():
        print("Error: Cannot open video stream")
        return
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("frame1", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()