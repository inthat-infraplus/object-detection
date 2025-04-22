import cv2
import ultralytics
from ultralytics import YOLO
import cvzone
import math

#cap = cv2.VideoCapture(0)  # laptop camera
cap = cv2.VideoCapture("Videos/road1.mp4")  # for video
#cap.set(3,1280)
#cap.set(4,720)

model = YOLO("weights/yolov8l.pt")  # Load the YOLOv8 model
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

while True:
    success,img = cap.read()  # Read a frame from the camera
    results = model.predict(source = img, stream=True,  )  # Perform inference on the frame
    for r in results:
        boxes = r.boxes
        for box in boxes:
            
            #Bounding Box
            x1,y1,x2,y2 = box.xyxy[0]
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)  # Convert coordinates to integers
            #cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)  # Draw the bounding box on the image
    
            w, h = x2-x1, y2-y1
            print(x1,y1,x2,y2)  # Print the coordinates of the bounding box
            cvzone.cornerRect(img,(x1,y1,w,h),l = 10)
            #Confidence Score
            conf = math.ceil((box.conf[0]*100))/100  # Calculate the confidence score 
            #Class Name
            cls =int(box.cls[0])
            currentClass = classNames[cls]
            
            if currentClass =="car" or currentClass =="truck" or currentClass =="bus" or currentClass =="motorbike" and conf > 0.3:
                cvzone.putTextRect(img,f'{currentClass} {conf}',(max(0,x1),max(35,y1)),
                               scale = 0.8,thickness=1,offset=3)
            
            
            
    cv2.imshow("Image",img)  # Display the image in a window
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Wait for 'q' key to exit
        break
