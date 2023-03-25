import numpy as np
import cv2
import homography_util
import os
import tensorflow as tf

path = "D:\\hpatches-sequences-release\\iv"


def get_dataset(train_path):
    dir_list = os.listdir(train_path)
    print("****dir:", dir_list)
    match_list_src = []
    match_list_dst = []
    h_list = []
    # Detect all keypoint
    for dir in dir_list:
        file_list_path = os.path.join(train_path, dir)
        file_list = os.listdir(file_list_path)
        print("path: ", file_list_path)
        print(file_list)
        kp_list = []
        desc_list = []
        for file in file_list[0:6]:
            # print(type(file), file)
            file_path = os.path.join(file_list_path, file)
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            # cv2.imshow("img", img)
            # cv2.waitKey(0)
            kp, desc = homography_util.get_kp_des(img)
            kp_list.append(kp)
            desc_list.append(desc)

        for file in file_list[6:]:
            file_path = os.path.join(file_list_path, file)
            # print(file_path)
            h = np.loadtxt(file_path)
            h_list.append(h.flatten()[:8])
            # find all match between each image

        for index in range(1, 6):
            src, desc = homography_util.get_match(kp_list[0], desc_list[0], kp_list[index], desc_list[index])
            src = src[:, 0, :]
            desc = desc[:, 0, :]
            match_list_src.append(src)
            match_list_dst.append(desc)

    return np.array(match_list_src), np.array(match_list_dst), np.array(h_list)


# file path
if __name__ == "__main__":
    data = np.load('data.npz')
    src = data['match_list_src']
    dst = data['match_list_dst']
    # h_list = data['h_list']
    # match_list_src, match_list_dst, h_list = get_dataset(path)
    # np.savez("data", match_list_src=match_list_src, match_list_dst= match_list_dst, h_list=h_list)
