from ultralytics import YOLO


def main():
    model = YOLO('yolov8s-obb.yaml')  # build from YAML and transfer weights
    model.train(data='dota8-obb.yaml', epochs=100, imgsz=1024, batch=4, workers=4)


if __name__ == '__main__':
    main()
