from ultralytics import YOLO
model = YOLO('runs/obb/train8/weights/best.pt')
results = model('datasets/images/val', save=True)
