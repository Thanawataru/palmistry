import torch
import numpy as np
import cv2

model = torch.hub.load('./yolov5', 'custom', path='yolov5s.pt', source='local')
model.conf = 0.20
model.iou = 0.20

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while cap.isOpened():

    timer = cv2.getTickCount()
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx = 1, fy = 1)

    #Make detections 
    results = model(frame)
    cv2.imshow('YOLO', np.squeeze(results.render()))
    cv2.waitKey(1)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
