import cv2
import numpy as np
from glob import glob
import random
import os


def get_train(path="E:/dataset/test2017/*.jpg", num=17700):
    rho = 32
    patch_size = 128
    width = 320
    height = 240
    loc_list = glob(path)
    print(len(loc_list))
    img_input = np.zeros((num, 128, 128, 2))  # images
    mat_output = np.zeros((num, 8))
    for i in range(num):
        # select random image from tiny training set
        # index = random.randint(0, len(loc_list) - 1)
        index = i
        # print(index)
        img_file_location = loc_list[index]
        gray_image = cv2.imread(img_file_location, cv2.IMREAD_GRAYSCALE)
        # cv2.imshow("gray", gray_image)
        # cv2.waitKey(0)
        # cv2.destroyWindow()
        gray_image = cv2.resize(gray_image, (width, height) )

        # create random point P within appropriate bounds
        # y = random.randint(rho, height - rho - patch_size)
        # x = random.randint(rho, width - rho - patch_size)
        x, y = 32, 32
        # define corners of image patch
        top_left_point = (x, y)
        bottom_left_point = (patch_size + x, y)
        bottom_right_point = (patch_size + x, patch_size + y)
        top_right_point = (x, patch_size + y)
        four_points = [top_left_point, bottom_left_point, bottom_right_point, top_right_point]
        perturbed_four_points = []
        for point in four_points:
            perturbed_four_points.append((point[0] + random.randint(-rho, rho), point[1] + random.randint(-rho, rho)))

        # compute H
        H = cv2.getPerspectiveTransform(np.float32(four_points), np.float32(perturbed_four_points))
        # print(H)
        # print(H.flatten())
        H_inverse = np.linalg.inv(H)
        inv_warped_image = cv2.warpPerspective(gray_image, H_inverse, (width, height))
        # warped_image = cv2.warpPerspective(gray_image, H, (width, height))

        # grab image patches
        original_patch = gray_image[y:y + patch_size, x:x + patch_size]
        warped_patch = inv_warped_image[y:y + patch_size, x:x + patch_size]
        # make into dataset
        # show_img = np.hstack((original_patch, warped_patch))
        training_image = np.dstack((original_patch, warped_patch))
        # cv2.imshow("pair", show_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        H_four_points = np.subtract(np.array(perturbed_four_points), np.array(four_points))
        img_input[i, :, :] = training_image
        mat_output[i, :] = H_four_points.reshape(-1)

    for n in range(0, 3):
        start = n * 5900
        end = start + 5900
        img = img_input[start:end]
        off = mat_output[start:end]
        # np.savez(f"./train_npz/img{n}", img=img, off=off)
        np.savez(f"./test_npz/img{n}", img=img, off=off)
    # return img_input, mat_output


def slice_npz(path):
    num = 5900
    start = 0
    data = np.load(path)
    images = data['img_input']
    offset = data['mat_output']
    for i in range(0, 20):
        start = i*num
        end = start+num
        img = images[start:end]
        off = offset[start:end]
        np.savez(f"./train_npz/img{i}", img=img, off=off)
        # np.savez(f"./offset_ npz/off{i}", off=off)


def data_loader(path, batch_size=64):
    while True:
        for npz in glob(os.path.join(path, '*.npz')):
            # Load pack into memory
            archive = np.load(npz)
            images = archive['img']
            offsets = archive['off']
            # Yield minibatch
            for i in range(0, len(offsets), batch_size):
                end_i = i + batch_size
                try:
                    batch_images = images[i:end_i]
                    batch_offsets = offsets[i:end_i]
                except IndexError:
                    continue
                # Normalize
                batch_images = (batch_images - 127.5) / 127.5
                batch_offsets = batch_offsets / 32.
                yield batch_images, batch_offsets


if __name__ == '__main__':
    # slice_npz("training_four_2017.npz")
    get_train()
    # data = np.load("train_npz/img0.npz")
    # inp = data['img']
    # out = data['off']
    print("****")
    # img_input, mat_output = get_train()
    # print("****")
    # np.savez("training_four", img_input=img_input, mat_output=mat_output)
