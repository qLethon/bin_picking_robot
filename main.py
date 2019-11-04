import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms
import os
from Network import AlexNet
from Network import train
from PIL import Image
import numpy as np
import argparse
import cv2
from serialTest.serialPackage import armCommunication


def transform_by4(img, points):
    """ copied from https://blanktar.jp/blog/2015/07/python-opencv-crop-box.html """
    """ 4点を指定してトリミングする。 """

    points = sorted(points, key=lambda x:x[1])  # yが小さいもの順に並び替え。
    top = sorted(points[:2], key=lambda x:x[0])  # 前半二つは四角形の上。xで並び替えると左右も分かる。
    bottom = sorted(points[2:], key=lambda x:x[0], reverse=True)  # 後半二つは四角形の下。同じくxで並び替え。
    points = np.array(top + bottom, dtype='float32')  # 分離した二つを再結合。

    width = max(np.sqrt(((points[0][0]-points[2][0])**2)*2), np.sqrt(((points[1][0]-points[3][0])**2)*2))
    height = max(np.sqrt(((points[0][1]-points[2][1])**2)*2), np.sqrt(((points[1][1]-points[3][1])**2)*2))

    dst = np.array([
            np.array([0, 0]),
            np.array([width-1, 0]),
            np.array([width-1, height-1]),
            np.array([0, height-1]),
            ], np.float32)

    trans = cv2.getPerspectiveTransform(points, dst)  # 変換前の座標と変換後の座標の対応を渡すと、透視変換行列を作ってくれる。
    return cv2.warpPerspective(img, trans, (int(width), int(height)))  # 透視変換行列を使って切り抜く。

def np_to_PIL(image):
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

def crop_image_along_line(image):
    blue, green, red = cv2.split(image)
    diff = np.where(red >= green, red - (green.astype(np.uint16) * 3 // 10).astype(np.uint8), 0)
    ret, thresh = cv2.threshold(diff, 150, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours.sort(key=cv2.contourArea, reverse=True)
    epsilon = 0.05 * cv2.arcLength(contours[0], True)
    approx = cv2.approxPolyDP(contours[0], epsilon, True)

    return transform_by4(image, approx[:, 0, :])

cam = cv2.VideoCapture(2)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)

def capture():
    # cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 4000)
    retval, frame = cam.read()
    if not retval:
        print('cannnot read')
    # return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return frame



def get_max_dir(directory_path):
    os.makedirs(directory_path, exist_ok=True)
    return max([0] + [int(d.name) for d in os.scandir(directory_path) if d.is_dir() and d.name.isdigit()])

def get_max_file(directory_path):
    os.makedirs(directory_path, exist_ok=True)
    return max([0] + [int(f.name.split('.')[0]) for f in os.scandir(directory_path) if f.is_file() and f.name.split('.')[0].isdigit()])

def crop_center(image, x, y, size):
    d = size // 2
    return image.crop((x - d, y - d, x + d + 1, y + d + 1))

def random_position():
    from random import randint
    return randint(0, 85), randint(0, 135)

def pick(y, x, indicator, arm):
    base_x = -120
    base_y = 75
    half_x_point = 135
    arm.send_position(indicator * half_x_point + base_x + x, base_y + y)
    print(indicator * half_x_point + base_x + x, base_y + y)
    while True:
        res = arm.read_one_byte()
        if res != 0:
            return res == 11

def main(model):
    INPUT_SIZE = 129
    BATCH = 256
    OBJECT_NUM = 10
    picked_count = 0
    indicator = 0
    
    arm = armCommunication('COM8',115200,0.01);
    save_dirctory = './models/' + str(get_max_dir('./models') + 1)
    os.makedirs(save_dirctory, exist_ok=True)
    net = AlexNet()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    net.to(device)
    if model is not None:
        net.load_state_dict(torch.load(model))
    net.eval()
    sigmoid = nn.Sigmoid()
    to_tensor = transforms.ToTensor()

    for i in range(int(1e6)):
        if i != 0 and (i == 100 or i % 500 == 0):
            model_save_path = os.path.join(save_dirctory, '{}.pth'.format(i))
            train(os.path.join(model_save_path))
            net.load_state_dict(torch.load(model_save_path))
            net.eval()

        if picked_count >= OBJECT_NUM:
            picked_count = 0
            indicator = (indicator + 1) & 1

        print('cap')
        image = np_to_PIL(crop_image_along_line(capture()))
        print('done')
        # TODO: crop and rotate an image alona a red rectangle

        dh = image.height // 255
        dw = image.width // 255
        P = np.ndarray(shape=(image.height, image.width), dtype=float)

        # with torch.no_grad():
        #     for h in range(INPUT_SIZE // 2, image.height - INPUT_SIZE // 2,  dh):
        #         for w in range(INPUT_SIZE // 2, image.width - INPUT_SIZE // 2, dw * BATCH):
        #             input_images = []
        #             BW = list(range(w, min(image.width - INPUT_SIZE // 2, w + BATCH * dw), dw))
        #             for bw in BW:
        #                 input_image = crop_center(image, bw, h, INPUT_SIZE)
        #                 input_images.append(to_tensor(input_image))
        #             outputs = sigmoid(net(torch.stack(input_images).to(device)))
        #             print(h, w)
                    
        #             for batch, bw in enumerate(BW):
        #                 P[h][bw] = outputs[batch]


        # max_h, max_w = np.unravel_index(np.argmax(P), P.shape)
        # print(np.unravel_index(np.argmax(P), P.shape))
        h, w = random_position()
        try:
            res = pick(h, w, indicator, arm)
        except Exception as e:
            print(e)
            continue
        picked_count += res
        print(picked_count)
        image_save_path = './images/{}/{}.jpg'.format(res, get_max_file('./images/{}'.format(res)) + 1)
        crop_center(image, w, h, INPUT_SIZE).save(image_save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str)
    args = parser.parse_args()
    main(args.model)