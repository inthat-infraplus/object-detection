import torch
import cv2
import numpy as np
import os
import csv  # For CSV file writing
from collections import defaultdict  # For counting class occurrences


# '''
# class_names = {
#                 0: "D00 (Longitudinal crack)", #หาเป็นความยาว
#                 1: "D10 (Lateral crack)", #หาเป็นความยาว
#                 2: "D20 (Alligator Crack)", #หาเป็นพื้นที่
#                 3: "D30 (Patching)",
#                 4: "D40 (Pothole)",
#                 5: "D43 (Crosswalk blur)",
#                 6: "D44 (White line blur)"
#                 }
#
# '''

# 1. Clone the YOLOv9 Repository (if you haven't already)
#  - You'll need to do this outside the notebook, in your terminal:
#    !git clone <YOLOv9_repository_url>
#    !cd YOLOv9_repository
#    # ... any other setup steps from the repo

# 2. Load the YOLOv9 Model
try:
    from ultralytics import YOLO  # If YOLOv9 uses Ultralytics
except ImportError:
    print("Ultralytics library not found. Please install it.")
    # !pip install ultralytics  # Uncomment if you want to install here (but better to do in terminal)
    exit()

# Load the model
model_path = './models/best.pt'  # Or the path to your YOLOv9 .pt file
model = YOLO(model_path)  # Let YOLO() handle loading
model.eval()  # Set to evaluation mode
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 3. Define Input and Output Directories
input_folders = ['./images/data_2002', './images/data_2013', './images/data_5009']
output_folders = ['./result/cal/result_data_2002', './result/cal/result_data_2013',
                  './result/cal/result_data_5009']  # Corrected output folder names
csv_output_folder = './result/csv_results_detections'  # Folder to save CSV files

# Create output directories if they don't exist
for out_folder in output_folders:
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
if not os.path.exists(csv_output_folder):
    os.makedirs(csv_output_folder)

# 4. Define Camera Setups
camera_setups = {
    'data_2002': {'width': 4.00, 'height': 5.00},  # meters
    'data_2013': {'width': 4.00, 'height': 5.00},  # meters
    'data_5009': {'width': 3.63, 'height': 2.49}  # meters
}

# 4. Process Images and Count Classes as well as color mapping
all_class_counts = defaultdict(int)
class_colors = {
    'D00 (Longitudinal crack)': (255, 0, 0),  # Blue
    'D10 (Lateral crack)': (0, 255, 0),  # Green
    'D20 (Alligator Crack)': (0, 0, 255),  # Red
    'D30 (Patching)': (255, 255, 0),  # Yellow
    'D40 (Pothole)': (255, 0, 255),  # Magenta
    'D43 (Crosswalk blur)': (0, 255, 255),  # Cyan
    'D44 (White line blur)': (128, 0, 128)  # Purple (ish)
}


def calculate_length(y1, y2, camera_setup):
    length_px = y2 - y1
    length_m = length_px * camera_setup['height'] / 640
    return float(length_m)


def calculate_width(x1, x2, camera_setup):
    width_px = x2 - x1
    width_m = width_px * camera_setup['width'] / 640
    return float(width_m)


def calculate_area(x1, x2, y1, y2, camera_setup):
    width_px = x2 - x1
    length_px = y2 - y1
    width_m = width_px * camera_setup['width'] / 640
    length_m = length_px * camera_setup['height'] / 640
    area_m = width_m * length_m
    return float(area_m)


for in_folder, out_folder in zip(input_folders, output_folders):
    print(f"Processing images from: {in_folder}")
    class_counts = defaultdict(int)
    camera_setup = camera_setups.get(os.path.basename(in_folder),
                                   camera_setups['data_2002'])  # Default to data_2002

    # Create a CSV file for each input folder
    csv_filename = os.path.join(csv_output_folder, f"{os.path.basename(in_folder)}_results.csv")
    with open(csv_filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        # Write the header row
        csv_writer.writerow(['Image Filename', 'Object ID', 'Class', 'Value', 'X1', 'Y1', 'X2', 'Y2', 'Confidence'])

        # Loop through all files in the input folder
        for filename in os.listdir(in_folder):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(in_folder, filename)
                img = cv2.imread(img_path)
                img_orig = img.copy()

                # --- Resize to 640x640 BEFORE processing ---
                img_resized_for_save = cv2.resize(img_orig, (640, 640))  # Resize for saving
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Preprocess
                img_resized = cv2.resize(img, (640, 640))
                img_normalized = img_resized / 255.0
                img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).float().unsqueeze(0).to(device)

                # Run Inference
                with torch.no_grad():
                    results = model(img_tensor)

                class_names = {
                    0: "D00",  # หาเป็นความยาว
                    1: "D10",  # หาเป็นความยาว
                    2: "D20",  # หาเป็นพื้นที่
                    3: "D30",
                    4: "D40",
                    5: "D43",
                    6: "D44"
                }

                # Process Results
                predictions = results[0].boxes.data.cpu().numpy()
                img_height, img_width = img_resized_for_save.shape[:2]
                object_id = 1
                # Visualize
                for pred in predictions:
                    xyxy = pred[:4]  # Extract the bounding box coordinates
                    conf = pred[4]  # Confidence score
                    cls = int(pred[5])  # Class ID (as integer)

                    # Check if cls is a valid *key* in class_names
                    if cls in class_names:
                        class_name = class_names[cls]
                    else:
                        class_name = f"Unknown Class: {cls}"
                    x1, y1, x2, y2 = map(int, xyxy)

                    # cv2.rectangle(img_resized_for_save, (x1, y1), (x2, y2),(255,0,0), 2)

                    if class_name == "D00":
                        value = calculate_length(y1, y2, camera_setup)
                        label = f'{class_name} id:{object_id} Length{value:.2f} m'
                        cv2.rectangle(img_resized_for_save, (x1, y1), (x2, y2), (255, 0, 0), 2)

                    elif class_name == "D10":
                        value = calculate_width(x1, x2, camera_setup)
                        label = f'{class_name} id:{object_id} Width {value:.2f} m'
                        cv2.rectangle(img_resized_for_save, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    elif class_name == "D20":
                        value = calculate_area(x1, x2, y1, y2, camera_setup)
                        label = f'{class_name} id:{object_id} Area {value:.2f} m2'
                        cv2.rectangle(img_resized_for_save, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    else:
                        continue  # Skip other classes

                    # --- Visualization Adjustments ---

                    box_color = class_colors.get(class_name, (255, 0, 255))

                    font_scale = 0.6
                    font_thickness = 1
                    font_color = (0, 0, 255)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    text_height = cv2.getTextSize(label, font, font_scale, font_thickness)[0][1]
                    text_width = cv2.getTextSize(label, font, font_scale, font_thickness)[0][0]

                    # Calculate label position - Handle all edge cases
                    label_y = y1 - 10  # Default: above the box
                    if label_y < text_height:
                        label_y = y2 + 15  # If too close to top, put below

                    label_x = x1  # Default: left-aligned with box
                    if label_x < 0:
                        label_x = 0  # If too close to left, start at left edge
                    elif label_x + text_width > img_width:
                        label_x = img_width - text_width  # If too close to right, align right edge

                    # if label_y > img_height - text_height:  # If too close to bottom
                    #  label_y = y1 - 10 if y1 - 10 > text_height else img_height - text_height - 10 # Try above, else bottom

                    cv2.putText(img_resized_for_save, label, (x1, label_y), font, font_scale, font_color,
                                font_thickness, cv2.LINE_AA)
                    # -------------------------------

                    class_counts[class_name] += 1
                    all_class_counts[class_name] += 1
                    # Write row to CSV file
                    csv_writer.writerow(
                        [filename, object_id, class_name, value, x1, y1, x2, y2, conf])  # Added x1,y1,x2,y2,conf
                    object_id += 1 # Increment object ID for next detection in the same image.

                # Save the visualized image to the output folder
                output_path = os.path.join(out_folder, filename)
                cv2.imwrite(output_path, cv2.cvtColor(img_resized_for_save, cv2.COLOR_RGB2BGR))  # Save the resized image

                print(f"  Processed and saved: {filename} to {out_folder}")
        print(f"Results for folder {in_folder} saved to {csv_filename}")

print("Processing complete!")
