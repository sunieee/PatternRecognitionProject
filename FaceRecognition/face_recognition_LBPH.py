import os
import numpy as np
import cv2
import time
import scipy

IMG_SIZE = (20, 20)


def load_img(file_name):
    """
    载入图像，统一尺寸，灰度化处理，直方图均衡化
    :param file_name: 图像文件名
    :return: 图像矩阵
    """
    t_img_mat = cv2.imread(file_name)  # 载入图像
    t_img_mat = cv2.resize(t_img_mat, IMG_SIZE)  # 统一尺寸
    t_img_mat = cv2.cvtColor(t_img_mat, cv2.COLOR_RGB2GRAY)  # 转化为灰度图
    img_mat = cv2.equalizeHist(t_img_mat)  # 直方图均衡
    return img_mat


def load_img_direct(img):
    t_img_mat = cv2.resize(img, IMG_SIZE)  # 统一尺寸
    img_mat = cv2.equalizeHist(t_img_mat)  # 直方图均衡
    return img_mat


class AlgorithmLbp(object):
    def __init__(self):
        self.table = {}
        self.ImgSize = IMG_SIZE
        self.BlockNum = 5
        self.count = 0

    def load_img_list(self, dir_name):
        """
        加载图像矩阵列表
        :param dir_name:文件夹路径
        :return: 包含最原始的图像矩阵的列表和标签矩阵
        """
        img_list = []
        label = []
        for parent, dir_names, file_names in os.walk(dir_name):
            for t_dir_name in dir_names:
                for sub_parent, sub_dir_name, sub_filenames in os.walk(parent + '/' + t_dir_name):
                    for file_name in sub_filenames:
                        if not file_name.endswith('.jpg'):
                            continue
                        if (file_name.endswith('.17.jpg')
                                or file_name.endswith('.18.jpg')
                                or file_name.endswith('.19.jpg')
                                or file_name.endswith('.20.jpg')):
                            continue
                        img_list.append(load_img(sub_parent + '/' + file_name))
                        label.append(sub_parent + '/' + file_name)
        return img_list, label

    def get_hop_counter(self, num):
        """
        计算二进制序列是否只变化两次
        :param num: 数字
        :return: 01变化次数
        """
        bin_num = bin(num)
        bin_str = str(bin_num)[2:]
        n = len(bin_str)
        if n < 8:
            bin_str = "0" * (8 - n) + bin_str
        n = len(bin_str)
        counter = 0
        for i in range(n):
            if i != n - 1:
                if bin_str[i + 1] != bin_str[i]:
                    counter += 1
            else:
                if bin_str[0] != bin_str[i]:
                    counter += 1
        return counter

    def get_table(self):
        """
        生成均匀对应字典
        :return: 均匀LBP特征对应字典
        """
        counter = 1
        for i in range(256):
            if self.get_hop_counter(i) <= 2:
                self.table[i] = counter
                counter += 1
            else:
                self.table[i] = 0
        return self.table

    def get_lbp_feature(self, img_mat):
        """
        计算LBP特征
        :param img_mat:图像矩阵
        :return: LBP特征图
        """
        m = img_mat.shape[0]
        n = img_mat.shape[1]
        neighbor = [0] * 8
        feature_map = np.mat(np.zeros((m, n)))
        t_map = np.mat(np.zeros((m, n)))
        for y in range(1, m - 1):
            for x in range(1, n - 1):
                neighbor[0] = img_mat[y - 1, x - 1]
                neighbor[1] = img_mat[y - 1, x]
                neighbor[2] = img_mat[y - 1, x + 1]
                neighbor[3] = img_mat[y, x + 1]
                neighbor[4] = img_mat[y + 1, x + 1]
                neighbor[5] = img_mat[y + 1, x]
                neighbor[6] = img_mat[y + 1, x - 1]
                neighbor[7] = img_mat[y, x - 1]
                center = img_mat[y, x]
                temp = 0
                for k in range(8):
                    temp += (neighbor[k] >= center) * (1 << k)
                feature_map[y, x] = self.table[temp]
                t_map[y, x] = temp
        feature_map = feature_map.astype('uint8')  # 数据类型转换为无符号8位型，如不转换则默认为float64位，影响最终效果
        self.count += 1
        return feature_map

    def get_hist(self, roi):
        """
        计算直方图
        :param roi:图像区域
        :return: 直方图矩阵
        """
        hist = cv2.calcHist([roi], [0], None, [59], [0, 256])  # 第四个参数是直方图的横坐标数目，经过均匀化降维后这里一共有59种像素
        return hist

    def compare(self, sampleImg, test_img):
        """
        比较函数，这里使用的是欧氏距离排序，也可以使用KNN，在此处更改
        :param sampleImg: 样本图像矩阵
        :param test_img: 测试图像矩阵
        :return: k2值
        """
        testFeatureMap = self.get_lbp_feature(test_img)
        sampleFeatureMap = self.get_lbp_feature(sampleImg)
        # 计算步长，分割整个图像为小块
        ystep = int(self.ImgSize[0] / self.BlockNum)
        xstep = int(self.ImgSize[1] / self.BlockNum)

        k2 = 0
        for y in range(0, self.ImgSize[0], ystep):
            for x in range(0, self.ImgSize[1], xstep):
                testroi = testFeatureMap[y:y + ystep, x:x + xstep]
                sampleroi = sampleFeatureMap[y:y + ystep, x:x + xstep]
                testHist = self.get_hist(testroi)
                sampleHist = self.get_hist(sampleroi)
                k2 += np.sum((sampleHist - testHist) ** 2) / np.sum((sampleHist + testHist))
        return k2

    def predict_faces_dataset(self, dir_path, test_path):
        """
        预测函数
        :param dir_path:样本图像文件夹路径
        :param test_path: 测试图像文件路径
        :return: 最相近图像名称
        """
        global acc_count
        self.table = self.get_table()

        test_img_list = []
        test_labels = []
        for parent, dir_names, file_names in os.walk(test_path):
            for t_dir_name in dir_names:
                for sub_parent, sub_dir_name, sub_filenames in os.walk(parent + '/' + t_dir_name):
                    for file_name in sub_filenames:
                        if not file_name.endswith('.jpg'):
                            continue
                        if (file_name.endswith('.17.jpg')
                                or file_name.endswith('.18.jpg')
                                or file_name.endswith('.19.jpg')
                                or file_name.endswith('.20.jpg')):
                            test_img_list.append(load_img(sub_parent + '/' + file_name))
                            test_labels.append(sub_parent.split('/')[-1])

        img_list, label = self.load_img_list(dir_path)

        acc_count = 0
        result_list = []
        time_1 = int(round(time.time() * 1000))
        for i in range(0, len(test_img_list)):
            test_img = test_img_list[i]
            k2_list = []
            for img in img_list:
                k2 = self.compare(img, test_img)
                k2_list.append(k2)
            result = label[np.argsort(k2_list)[0]]
            result = result.split('/')[-2]
            result_list.append(result)
            acc_count = acc_count + 1 if result == test_labels[i] else acc_count
            print("Predicted: " + result + ' -- Labels: ' + test_labels[i])
        time_2 = int(round(time.time() * 1000))
        compare_time = time_2 - time_1

        print("Comparing Finished in: ", compare_time)
        print("---- Accuracy is: ", acc_count / len(result_list), " ----")

        return result_list

    def process_mat(self, file_path):
        data = scipy.io.loadmat(file_path)
        is_test = data['isTest']
        figures = data['fea']
        labels = data['gnd']

        test_img_list = []
        test_labels = []
        img_list = []
        label = []

        for i in range(0, len(is_test)):
            if is_test[i] == 1:
                test_img_list.append(figures[i].reshape(64, 64))
                test_labels.append(labels[i])
            else:
                img_list.append(figures[i].reshape(64, 64))
                label.append(labels[i])

        return test_img_list, test_labels, img_list, label

    def predict_pie_dataset(self, path):
        global acc_count
        self.table = self.get_table()

        test_img_list = []
        test_labels = []
        img_list = []
        label = []

        if os.path.isfile(path):
            test_img_list, test_labels, img_list, label = self.process_mat(path)
        elif os.path.isdir(path):
            for root, dirs, files in os.walk(path):
                for file in files:
                    returns = self.process_mat(os.path.join(root, file))
                    test_img_list, test_labels, img_list, label = test_img_list + returns[0], test_labels + returns[1], img_list + returns[2], label + returns[3]

        acc_count = 0
        result_list = []
        time_1 = int(round(time.time() * 1000))
        for i in range(0, len(test_img_list)):
            test_img = test_img_list[i]
            k2_list = []
            for img in img_list:
                k2 = self.compare(load_img_direct(img), load_img_direct(test_img))
                k2_list.append(k2)
            result = label[np.argsort(k2_list)[0]]
            result_list.append(result)
            acc_count = acc_count + 1 if result == test_labels[i] else acc_count
            print("Predicted: " + result + ' -- Labels: ' + test_labels[i])
        time_2 = int(round(time.time() * 1000))
        compare_time = time_2 - time_1

        print("Comparing Finished in: ", compare_time)
        print("---- Accuracy is: ", acc_count / len(result_list), " ----")

        return result_list


if __name__ == '__main__':
    result_list = AlgorithmLbp().predict_faces_dataset('../dataset/faces/grimace', '../dataset/faces/grimace')
    # result_list = AlgorithmLbp().predict_pie_dataset('../dataset/PIE dataset')
