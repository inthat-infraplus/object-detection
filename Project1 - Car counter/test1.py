from ultralytics import YOLO

# Load an official or custom model
model = YOLO("../weights/yolo11l.pt")  # Load an official Detect model


# Perform tracking with the model
#results = model.track("../Videos/road4.mp4", show=True)  # Tracking with default tracker
results = model.track("../Videos/road4.mp4", show=True, tracker="bytetrack.yaml")  # with ByteTrack