import torch
import numpy as np
import cv2
from roboflow import Roboflow

# สร้างวัตถุในคลาส Roboflow เพื่อเชื่อมต่อกับ Roboflow API
rf = Roboflow(api_key="AWLFKZigjwPXIuSYYjZE")

# ระบุพื้นที่ทำงาน (Workspace) และโปรเจค (Project) ที่ต้องการดาวน์โหลดชุดข้อมูล
project = rf.workspace("kamon").project("kamontrain")

# ระบุเวอร์ชันของโปรเจคที่ต้องการดาวน์โหลดและดาวน์โหลดชุดข้อมูล (Dataset)
dataset = project.version(1).download("yolov5")

# โหลดโมเดล YOLOv5 ที่ดาวน์โหลดมาจาก Roboflow โดยใช้ไฟล์ที่อยู่ใน "yolov5s.pt"
model = torch.hub.load('./yolov5', 'custom', path=dataset, source='local')

model.conf = 0.60
model.iou = 0.45

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while cap.isOpened():
    timer = cv2.getTickCount()
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=1, fy=1)

    #Make detections 
    results = model(frame)
    cv2.imshow('YOLO', np.squeeze(results.render()))
    cv2.waitKey(1)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
