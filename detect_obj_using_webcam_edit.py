import torch
import numpy as np
import cv2

model = torch.hub.load('./yolov5', 'custom', path='yolov5s.pt', source='local')
model.conf = 0.4
#model.iou = 0.45

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while cap.isOpened():

    timer = cv2.getTickCount()
    ret, frame = cap.read()
    res = frame.copy() 
    #frame = cv2.resize(frame, None, fx = 1, fy = 1)

    # Make detections 
    results = model(frame)
    xywh = results.xywh[0] # Get Objs bounding box
    objs = xywh.numpy() #Convert bounding box to numpy
    obj_list = model.names

    class_num = 0
    if objs.shape[0] != 0: # Check number of detected objs > 0
        for obj in objs:
            detected_obj = obj_list[int(obj[5])]
            if detected_obj == 'person':
                cx, cy, w, h, conf_val = obj.astype(int)[:5]
                #cv2.circle(res, (cx,cy), 10, (0, 0, 255))
                res = cv2.rectangle(res, (int(cx - w/2), int(cy - h/2)), (int(cx + w/2), int(cy + h/2)), (255, 0, 0),  2)             

    cv2.imshow('YOLO', res)
    cv2.waitKey(1)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
