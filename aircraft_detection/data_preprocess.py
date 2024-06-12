import os
import cv2
import numpy as np
import random


def get_random_subimages(image, mask, num_subimages, min_size, max_size):
    h, w, _ = image.shape
    subimages = []

    for _ in range(num_subimages):
        while True:
            sub_h = random.randint(min_size, max_size)
            sub_w = random.randint(min_size, max_size)

            if sub_h > h or sub_w > w:
                continue

            top_left_x = random.randint(0, w - sub_w)
            top_left_y = random.randint(0, h - sub_h)

            bottom_right_x = top_left_x + sub_w
            bottom_right_y = top_left_y + sub_h

            sub_image = image[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
            sub_mask = mask[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

            if np.all(sub_mask == 255):
                subimages.append(sub_image)
                break

    return subimages


if __name__ == '__main__':
    predata_dir = './preDataset'
    if not os.path.exists(predata_dir):
        os.mkdir(predata_dir)

    train_dir = './dataset/train'
    images = os.listdir(os.path.join(train_dir, 'images'))
    labels = os.listdir(os.path.join(train_dir, 'labelTxt'))

    for image, label in zip(images, labels):
        rectangles = []
        image_path = os.path.join(train_dir, 'images', image)
        label_path = os.path.join(train_dir, 'labelTxt', label)
        image_arr = cv2.imread(image_path)
        with open(label_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                aircraft_coordinate = [float(x) for x in line[:len(line) - 12].split(' ')]
                pts_src = np.array([[aircraft_coordinate[0], aircraft_coordinate[1]],
                                    [aircraft_coordinate[2], aircraft_coordinate[3]],
                                    [aircraft_coordinate[4], aircraft_coordinate[5]],
                                    [aircraft_coordinate[6], aircraft_coordinate[7]]], dtype=np.float32)
                width_top = np.linalg.norm(pts_src[0] - pts_src[1])
                width_bottom = np.linalg.norm(pts_src[3] - pts_src[2])
                height_left = np.linalg.norm(pts_src[0] - pts_src[3])
                height_right = np.linalg.norm(pts_src[1] - pts_src[2])
                width = int(max(width_top, width_bottom))
                height = int(max(height_left, height_right))
                pts_dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
                M = cv2.getPerspectiveTransform(pts_src, pts_dst)
                warped = cv2.warpPerspective(image_arr, M, (width, height))
                cv2.imwrite(os.path.join(predata_dir, '1', image), warped)

                rectangles.append(np.array([[aircraft_coordinate[0], aircraft_coordinate[1]],
                                            [aircraft_coordinate[2], aircraft_coordinate[3]],
                                            [aircraft_coordinate[4], aircraft_coordinate[5]],
                                            [aircraft_coordinate[6], aircraft_coordinate[7]]], dtype=np.int32))

        height, width = image_arr.shape[:2]
        mask = np.ones((height, width), np.uint8) * 255
        for rectangle in rectangles:
            cv2.fillPoly(mask, [rectangle], 0)
        masked_image = cv2.bitwise_and(image_arr, image_arr, mask=mask)

        min_size = 50
        max_size = 150
        num_subimages = 20
        subimages = get_random_subimages(masked_image, mask, num_subimages, min_size, max_size)

        for idx, sub_image in enumerate(subimages):
            cv2.imwrite(os.path.join(predata_dir, '0', image + str(idx) + '.png'), sub_image)

