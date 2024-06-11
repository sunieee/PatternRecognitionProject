from mmdet.apis import init_detector, inference_detector
import mmcv
import torch
import os
import numpy as np

# Initialize the model
config_file = 'oriented_rcnn_r50_fpn_1x_dota_le90.py'
checkpoint_file = 'oriented_rcnn_r50_fpn_1x_dota_le90-6d2b2ce0.pth'
device = 'cuda:0'

model = init_detector(config_file, checkpoint_file, device=device)

# Directory for saving results
output_dir = 'results/'
os.makedirs(output_dir, exist_ok=True)

# Inference on the test dataset
img_dir = 'data/test/images/'
result_files = []

for img_name in os.listdir(img_dir):
    img_path = os.path.join(img_dir, img_name)
    result = inference_detector(model, img_path)
    
    # Extract the results for the "plane" class (assuming it is class 0)
    planes = result[0]  # Assuming "plane" is the first class
    
    # Save the results
    output_file = os.path.join(output_dir, img_name.replace('.png', '.txt'))
    result_files.append(output_file)
    with open(output_file, 'w') as f:
        for plane in planes:
            if plane[4] > 0.05:  # Filter out low-confidence detections
                bbox = plane[:4]
                score = plane[4]
                f.write(f"{bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]} {score}\n")

print(f"Results saved to {output_dir}")
