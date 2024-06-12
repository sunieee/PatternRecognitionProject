from mmdet.apis import init_detector, inference_detector
# 这个import很重要，否则KeyError: 'OrientedRCNN is not in the models registry'
import mmrotate
import mmcv
import torch
import os
import numpy as np
from tqdm import tqdm
import math

# 添加参数，指定配置文件和模型文件，输出目录
import argparse
parser = argparse.ArgumentParser(description='Inference on the test dataset')
parser.add_argument('--config', type=str, default='oriented_rcnn_r50_fpn_1x_dota_le90.py', help='Config file')
parser.add_argument('--checkpoint', type=str, default='oriented_rcnn_r50_fpn_1x_dota_le90-6d2b2ce0.pth', help='Checkpoint file')
parser.add_argument('--output-label-dir', type=str, default='results/pretrainlabel', help='Directory for saving label files')
parser.add_argument('--output-image-dir', type=str, default='results/pretrainimages', help='Directory for saving images files')
args = parser.parse_args()

# Initialize the model
# config_file = 'oriented_rcnn_r50_fpn_1x_dota_le90.py'
# checkpoint_file = 'oriented_rcnn_r50_fpn_1x_dota_le90-6d2b2ce0.pth'
config_file = args.config
checkpoint_file = args.checkpoint
output_label_dir = args.output_label_dir
output_image_dir = args.output_image_dir
device = 'cuda:0'

model = init_detector(config_file, checkpoint_file, device=device)
os.makedirs(output_label_dir, exist_ok=True)
os.makedirs(output_image_dir, exist_ok=True)

# Inference on the test dataset
img_dir = 'data/test/images/'
result_files = []

# Function to convert oriented bbox to polygon
def bbox_to_polygon(cx, cy, w, h, angle):
    theta = angle * math.pi / 180
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    
    w_half = w / 2
    h_half = h / 2
    
    points = [
        (-w_half, -h_half),
        (w_half, -h_half),
        (w_half, h_half),
        (-w_half, h_half)
    ]
    
    polygon = []
    for x, y in points:
        x_rot = x * cos_theta - y * sin_theta + cx
        y_rot = x * sin_theta + y * cos_theta + cy
        polygon.append((x_rot, y_rot))
    
    return polygon

print("Performing inference...")

for img_name in tqdm(os.listdir(img_dir)):
    img_path = os.path.join(img_dir, img_name)
    result = inference_detector(model, img_path)
    
    # Extract the results for the "plane" class (assuming it is class 0)
    planes = result[0]  # Assuming "plane" is the first class
    
    # Save the results in label format
    output_file = os.path.join(output_label_dir, img_name.replace('.png', '.txt'))
    result_files.append(output_file)
    with open(output_file, 'w') as f:
        for plane in planes:
            if plane[5] > 0.7:  # Filter out low-confidence detections
                cx, cy, w, h, angle, score = plane
                polygon = bbox_to_polygon(cx, cy, w, h, angle * 180 / math.pi)
                polygon_str = ' '.join(f"{coord[0]:.1f} {coord[1]:.1f}" for coord in polygon)
                # 默认difficult 取0
                f.write(f"{polygon_str} Airplane 0\n")
    
    # Save the result image
    output_image_path = os.path.join(output_image_dir, img_name)
    model.show_result(img_path, result, out_file=output_image_path)

print(f"Results saved to {output_label_dir} and {output_image_dir}")