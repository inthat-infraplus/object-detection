from ultralytics import YOLO
import cv2

model = YOLO("weights/yolov8l.pt")  # Load the YOLOv8 model
results = model("Images/image1.jpg",show=True)  # Perform inference on the image
cv2.waitKey(0)  # Wait for a key press to close the image window