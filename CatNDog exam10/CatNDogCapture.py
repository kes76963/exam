import cv2
import time

capture =cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
flag = True

while flag :
    ret, frame = capture.read()
    cv2.imshow('VideoFrame', frame)
    time.sleep(0.1)
    print('capture')
    key = cv2.waitKey(33)
    if key == 27 :

        flag = False

capture.release() # 카메라 놔주기 / 중복 접근 막기 위해서
cv2.destroyALLWindows() # 종료 시키기