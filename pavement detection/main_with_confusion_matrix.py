import matplotlib
matplotlib.use('Agg')  # Try 'Agg' backend first

import torch
import cv2
import numpy as np
import os
import csv
from collections import defaultdict
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from ultralytics import YOLO
except ImportError:
    print("Ultralytics library not found. Please install it.")
    exit()

# Load the model
model_path = './models/best.pt'
model = YOLO(model_path)
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Define Input and Output Directories
input_folders = ['./images/data_2002', './images/data_2013', './images/data_5009']
output_folders = ['./result/cal/result_data_2002', './result/cal/result_data_2013',
                  './result/cal/result_data_5009']
csv_output_folder = './result/csv_results_detections'
confusion_matrix_output = './result/confusion_matrix'
annotations_dir = './result/annotations'

# Create output directories if they don't exist
for out_folder in output_folders:
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
if not os.path.exists(csv_output_folder):
    os.makedirs(csv_output_folder)
if not os.path.exists(confusion_matrix_output):
        os.makedirs(confusion_matrix_output)
if not os.path.exists(annotations_dir):
    os.makedirs(annotations_dir)

# 4. Define Camera Setups
camera_setups = {
    'data_2002': {'width': 4.00, 'height': 5.00},
    'data_2013': {'width': 4.00, 'height': 5.00},
    'data_5009': {'width': 3.63, 'height': 2.49}
}

# Class names mapping for consistency
class_names_map = {
    0: "D00",
    1: "D10",
    2: "D20",
    3: "D30",
    4: "D40",
    5: "D43",
    6: "D44"
}

# 4. Process Images and Count Classes as well as color mapping
all_class_counts = defaultdict(int)
class_colors = {
    'D00 (Longitudinal crack)': (255, 0, 0),
    'D10 (Lateral crack)': (0, 255, 0),
    'D20 (Alligator Crack)': (0, 0, 255),
    'D30 (Patching)': (255, 255, 0),
    'D40 (Pothole)': (255, 0, 255),
    'D43 (Crosswalk blur)': (0, 255, 255),
    'D44 (White line blur)': (128, 0, 128)
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

all_predictions = []
all_ground_truth = []


def get_ground_truth(image_filename):
    base_name, ext = os.path.splitext(image_filename)
    annotation_filename = os.path.join(annotations_dir, base_name + '.txt')

    ground_truth_labels = []
    if os.path.exists(annotation_filename):
        with open(annotation_filename, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    try:
                        class_id = int(parts[0])
                        if class_id in class_names_map:
                            ground_truth_labels.append(class_names_map[class_id])
                    except ValueError:
                        print(f"Error reading class ID from line: {line.strip()} in file: {annotation_filename}")
                        continue
    return ground_truth_labels



for in_folder, out_folder in zip(input_folders, output_folders):
    print(f"Processing images from: {in_folder}")
    class_counts = defaultdict(int)
    camera_setup = camera_setups.get(os.path.basename(in_folder),
                                        camera_setups['data_2002'])

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

                ground_truth_labels = get_ground_truth(filename)
                local_predictions = []

                # --- Resize to 640x640 BEFORE processing ---
                img_resized_for_save = cv2.resize(img_orig, (640, 640))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Preprocess
                img_resized = cv2.resize(img, (640, 640))
                img_normalized = img_resized / 255.0
                img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).float().unsqueeze(0).to(device)

                # Run Inference
                with torch.no_grad():
                    results = model(img_tensor)

                # Process Results
                predictions = results[0].boxes.data.cpu().numpy()
                img_height, img_width = img_resized_for_save.shape[:2]
                object_id = 1

                # Prepare to write to annotation file
                base_name = os.path.splitext(filename)[0]
                annotation_filename = os.path.join(annotations_dir, base_name + '.txt')
                with open(annotation_filename, 'w') as annotation_file:
                    # Visualize
                    for pred in predictions:
                        xyxy = pred[:4]
                        conf = pred[4]
                        cls = int(pred[5])

                        # Use the consistent class names from class_names_map
                        if cls in class_names_map:
                            class_name_base = class_names_map[cls]
                            class_name_display = [k for k in class_colors if k.startswith(class_name_base)][0] if class_name_base in [k.split(' ')[
                                0] for k in class_colors] else class_name_base
                        else:
                            class_name_base = f"Unknown Class: {cls}"
                            class_name_display = class_name_base
                        x1, y1, x2, y2 = map(int, xyxy)

                        value = None
                        label = class_name_display

                        if class_name_base == "D00":
                            value = calculate_length(y1, y2, camera_setup)
                            label = f'{class_name_display} id:{object_id} Length {value:.2f} m'
                            cv2.rectangle(img_resized_for_save, (x1, y1), (x2, y2),
                                          class_colors['D00 (Longitudinal crack)'], 2)
                        elif class_name_base == "D10":
                            value = calculate_width(x1, x2, camera_setup)
                            label = f'{class_name_display} id:{object_id} Width {value:.2f} m'
                            cv2.rectangle(img_resized_for_save, (x1, y1), (x2, y2),
                                          class_colors['D10 (Lateral crack)'], 2)
                        elif class_name_base == "D20":
                            value = calculate_area(x1, x2, y1, y2, camera_setup)
                            label = f'{class_name_display} id:{object_id} Area {value:.2f} m2'
                            cv2.rectangle(img_resized_for_save, (x1, y1), (x2, y2),
                                          class_colors['D20 (Alligator Crack)'], 2)
                        elif class_name_base in [k.split(' ')[0] for k in class_colors]:
                            color_key = [k for k in class_colors if k.startswith(class_name_base)][0]
                            cv2.rectangle(img_resized_for_save, (x1, y1), (x2, y2), class_colors[color_key], 2)

                        if class_name_base in [k.split(' ')[0] for k in class_colors]:
                            color_key = [k for k in class_colors if k.startswith(class_name_base)][0]
                            box_color = class_colors[color_key]
                            font_scale = 0.6
                            font_thickness = 1
                            font_color = (0, 0, 255)
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            text_height = cv2.getTextSize(label, font, font_scale, font_thickness)[0][1]
                            text_width = cv2.getTextSize(label, font, font_scale, font_thickness)[0][0]

                            # Calculate label position - Handle all edge cases
                            label_y = y1 - 10
                            if label_y < text_height:
                                label_y = y2 + 15

                            label_x = x1
                            if label_x < 0:
                                label_x = 0
                            elif label_x + text_width > img_width:
                                label_x = img_width - text_width

                            cv2.putText(img_resized_for_save, label, (x1, label_y), font, font_scale, font_color,
                                        font_thickness, cv2.LINE_AA)

                            class_counts[class_name_base] += 1
                            all_class_counts[class_name_base] += 1
                            local_predictions.append(class_name_base)

                            # Convert bounding box to normalized coordinates
                            x_center = (x1 + (x2 - x1) / 2) / img_width
                            y_center = (y1 + (y2 - y1) / 2) / img_height
                            width_norm = (x2 - x1) / img_width
                            height_norm = (y2 - y1) / img_height

                            # Write to annotation file
                            annotation_line = f"{cls} {x_center:.6f} {y_center:.6f} {width_norm:.6f} {height_norm:.6f}\n"
                            annotation_file.write(annotation_line)

                            # Write row to CSV file
                            csv_writer.writerow(
                                [filename, object_id, class_name_display, value, x1, y1, x2, y2, conf])
                            object_id += 1

                # Basic matching
                min_len = min(len(ground_truth_labels), len(local_predictions))
                print(f"Filename: {filename}, len(ground_truth_labels): {len(ground_truth_labels)}, len(local_predictions): {len(local_predictions)}")
                if min_len > 0:
                    for i in range(min_len):
                        all_predictions.append(local_predictions[i])
                        all_ground_truth.append(ground_truth_labels[i])
                else:
                    print(f"Skipping matching for {filename} because either ground_truth_labels or local_predictions is empty")

                # Save the visualized image
                output_path = os.path.join(out_folder, filename)
                cv2.imwrite(output_path, cv2.cvtColor(img_resized_for_save, cv2.COLOR_RGB2BGR))

                print(f"  Processed and saved: {filename} to {out_folder}")
        print(f"Results for folder {in_folder} saved to {csv_filename}")

print("Processing complete!")


# --- Generate and Save Confusion Matrix using ConfusionMatrixDisplay ---
if all_ground_truth:
    unique_classes = sorted(list(set(all_ground_truth) | set(all_predictions)))
    cm = confusion_matrix(all_ground_truth, all_predictions, labels=unique_classes)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=unique_classes)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    confusion_matrix_path = os.path.join(confusion_matrix_output, 'confusion_matrix_display.png')
    plt.savefig(confusion_matrix_path)
    plt.close()

    print(f"Confusion matrix saved to {confusion_matrix_path}")
else:
    print("No ground truth data available to generate the confusion matrix.")
