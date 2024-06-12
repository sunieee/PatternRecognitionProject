import json
import os
import random

def split_coco_annotation(coco_ann_file, train_ratio=0.8, seed=42):
    random.seed(seed)
    
    with open(coco_ann_file, 'r') as f:
        coco_data = json.load(f)
    
    images = coco_data['images']
    annotations = coco_data['annotations']
    
    random.shuffle(images)
    
    train_size = int(len(images) * train_ratio)
    train_images = images[:train_size]
    val_images = images[train_size:]
    
    train_image_ids = {img['id'] for img in train_images}
    val_image_ids = {img['id'] for img in val_images}
    
    train_annotations = [ann for ann in annotations if ann['image_id'] in train_image_ids]
    val_annotations = [ann for ann in annotations if ann['image_id'] in val_image_ids]
    
    train_data = {
        'images': train_images,
        'annotations': train_annotations,
        'categories': coco_data['categories']
    }
    
    val_data = {
        'images': val_images,
        'annotations': val_annotations,
        'categories': coco_data['categories']
    }
    
    with open('data/train_val/train_coco_ann.json', 'w') as f:
        json.dump(train_data, f)
    
    with open('data/train_val/val_coco_ann.json', 'w') as f:
        json.dump(val_data, f)

if __name__ == '__main__':
    split_coco_annotation('data/train_val/train_val_coco_ann.json')
