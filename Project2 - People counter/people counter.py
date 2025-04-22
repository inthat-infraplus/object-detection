import cv2
import ultralytics
from ultralytics import YOLO
import cvzone
import math
from sort import *
import numpy as np

#cap = cv2.VideoCapture(0)  # laptop camera
cap = cv2.VideoCapture("../Videos/esca1.mp4")  # for video
#cap.set(3,1280)
#cap.set(4,720)

model = YOLO("../weights/yolo11l.pt")  # Load the YOLOv11 model
classNames = {0: 'person',
 1: 'bicycle',
 2: 'car',
 3: 'motorcycle',
 4: 'airplane',
 5: 'bus',
 6: 'train',
 7: 'truck',
 8: 'boat',
 9: 'traffic light',
 10: 'fire hydrant',
 11: 'stop sign',
 12: 'parking meter',
 13: 'bench',
 14: 'bird',
 15: 'cat',
 16: 'dog',
 17: 'horse',
 18: 'sheep',
 19: 'cow',
 20: 'elephant',
 21: 'bear',
 22: 'zebra',
 23: 'giraffe',
 24: 'backpack',
 25: 'umbrella',
 26: 'handbag',
 27: 'tie',
 28: 'suitcase',
 29: 'frisbee',
 30: 'skis',
 31: 'snowboard',
 32: 'sports ball',
 33: 'kite',
 34: 'baseball bat',
 35: 'baseball glove',
 36: 'skateboard',
 37: 'surfboard',
 38: 'tennis racket',
 39: 'bottle',
 40: 'wine glass',
 41: 'cup',
 42: 'fork',
 43: 'knife',
 44: 'spoon',
 45: 'bowl',
 46: 'banana',
 47: 'apple',
 48: 'sandwich',
 49: 'orange',
 50: 'broccoli',
 51: 'carrot',
 52: 'hot dog',
 53: 'pizza',
 54: 'donut',
 55: 'cake',
 56: 'chair',
 57: 'couch',
 58: 'potted plant',
 59: 'bed',
 60: 'dining table',
 61: 'toilet',
 62: 'tv',
 63: 'laptop',
 64: 'mouse',
 65: 'remote',
 66: 'keyboard',
 67: 'cell phone',
 68: 'microwave',
 69: 'oven',
 70: 'toaster',
 71: 'sink',
 72: 'refrigerator',
 73: 'book',
 74: 'clock',
 75: 'vase',
 76: 'scissors',
 77: 'teddy bear',
 78: 'hair drier',
 79: 'toothbrush'}

mask = cv2.imread("esca mask.png")  # Load the mask image

#tracking
tracker = Sort(max_age=20,min_hits=3,iou_threshold=0.3)
#limit = [1100,500,1370 ,600]
limit = [1100,650,1420 ,900]  
totalCount = []

limitsUp = [1100,500,1370 ,600]
limitsDown = [1100,650,1420 ,900]  
totalCountUp = []
totalCountDown = []

while True:
    success,img = cap.read()  # Read a frame from the camera
    imgRegion = cv2.bitwise_and(img,mask)  # Apply the mask to the image
    results = model.predict(source = imgRegion, stream=True)  # Perform inference on the frame
    detections = np.empty((0,5))
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            
            #Bounding Box
            x1,y1,x2,y2 = box.xyxy[0]
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)  # Convert coordinates to integers
    
            w, h = x2-x1, y2-y1
            print(x1,y1,x2,y2)  # Print the coordinates of the bounding box
            #Confidence Score
            conf = math.ceil((box.conf[0]*100))/100  # Calculate the confidence score 
            #Class Name
            cls =int(box.cls[0])
            currentClass = classNames[cls]
            
            if currentClass == 'person' and conf > 0.3:
                cvzone.putTextRect(img,f'{currentClass} {conf}',(max(0,x1),max(100,y1)),scale = 0.8,thickness=1,offset=3)
                cvzone.cornerRect(img,(x1,y1,w,h),l = 10,rt=2)
                #store array for tracking
                currentArray = np.array([x1,y1,x2,y2,conf])
                detections = np.vstack((detections,currentArray))
    
    resultsTracker = tracker.update(detections) 
    cv2.line(img,(limitsUp[0],limitsUp[1]),(limitsUp[2],limitsUp[3]),(0,0,255),5)  # Draw a line on the image
    cv2.line(img,(limitsDown[0],limitsDown[1]),(limitsDown[2],limitsDown[3]),(255,0,0),5)  # Draw a line on the image
    for result in resultsTracker:
        x1,y1,x2,y2,id = result
        x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
        w, h = x2-x1, y2-y1 #โง่
        print(result)
        cvzone.cornerRect(img,(x1,y1,w,h),l = 9,rt=2,colorR=(0,0,255))
        cvzone.putTextRect(img,f'{int(id)}',(max(0,x1),max(35,y1)),scale = 1,thickness=1,offset=10)
        
        cx,cy = (x1+w//2),(y1+h//2)  # Calculate the center of the bounding box
        cv2.circle(img,(cx,cy),5,(0,0,255),cv2.FILLED)
        if limitsUp[0] < cx < limitsUp[2] and limitsUp[1]-70 < cy < limitsUp[1]+70:
            if totalCountUp.count(id) ==0:
                totalCountUp.append(id)
                cv2.line(img,(limitsUp[0],limitsUp[1]),(limitsUp[2],limitsUp[3]),(0,255,0),5)
        
        if limitsDown[0] < cx < limitsDown[2] and limitsDown[1]-70 < cy < limitsDown[1]+70:
            if totalCountDown.count(id) ==0:
                totalCountDown.append(id)
                cv2.line(img,(limitsDown[0],limitsDown[1]),(limitsDown[2],limitsDown[3]),(0,255,0),5)
        # cvzone.putTextRect(img, f'Count : {len(set(totalCountUp))}', (50, 50))
    cv2.putText(img,str(len(totalCountUp)),(255,100),cv2.FONT_HERSHEY_PLAIN,5,(139,195,75),5) 
    cv2.putText(img,str(len(totalCountDown)),(400,100),cv2.FONT_HERSHEY_PLAIN,5,(250,0,0),5)    
    cv2.imshow("Image",img)  # Display the image in a window
    #cv2.imshow("Image region",imgRegion)  # Display the masked image in a window
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Wait for 'q' key to exit
       break
                  
