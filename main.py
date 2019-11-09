import torch
import time
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
from collections import deque
from valid import probability_to_green_image_array


ARM_RANGE_HEIGHT = 87
ARM_RANGE_WIDTH = 250
BASE_X = -125
BASE_Y = 68

def update_points(points):
    pointsOldDataFile = open('pointsOldData.csv','w')
    for _point in points:
        pointLineString = str(_point[0])+","+str(_point[1]) + "\n"
        pointsOldDataFile.write(pointLineString)
    pointsOldDataFile.close()

def read_savedPoints():
    points = []
    with open('pointsOldData.csv','r') as f:
        for pointLineString_fromFile in f:
            pointStrings = pointLineString_fromFile.split(",")
            points.append([float(p) for p in pointStrings])
    return points

def transform_by4(img, points, width, height):
    """ copied from https://blanktar.jp/blog/2015/07/python-opencv-crop-box.html """
    """ 4点を指定してトリミングする。 """
    if len(points) != 4: #頂点の数が4つでないなら古いデータを使う
        print("ないんじゃ～～")
        points = read_savedPoints()
    else:                   #頂点の数が4つなら古いデータ更新
        update_points(points)

    points = sorted(points, key=lambda x:x[1])  # yが小さいもの順に並び替え。
    top = sorted(points[:2], key=lambda x:x[0])  # 前半二つは四角形の上。xで並び替えると左右も分かる。
    bottom = sorted(points[2:], key=lambda x:x[0], reverse=True)  # 後半二つは四角形の下。同じくxで並び替え。
    points = np.array(top + bottom, dtype='float32')  # 分離した二つを再結合。
    dst = np.array([
            np.array([0, 0]),
            np.array([width-1, 0]),
            np.array([width-1, height-1]),
            np.array([0, height-1]),
            ], np.float32)
    trans = cv2.getPerspectiveTransform(points, dst)  # 変換前の座標と変換後の座標の対応を渡すと、透視変換行列を作ってくれる。(射影行列では？)
    return cv2.warpPerspective(img, trans, (int(width), int(height)))  #ここで影を指定のサイズで受け取る

def np_to_PIL(image):
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

def crop_image_along_line(image, width, height):
    blue, green, red = cv2.split(image)
    diff = np.where(green >= red, green - (red.astype(np.uint16) * 10 // 10).astype(np.uint8), 0)
    ret, thresh = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY)
    kernel = np.ones((50,50),np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours.sort(key=cv2.contourArea, reverse=True)
    epsilon = 0.05 * cv2.arcLength(contours[0], True)
    approx = cv2.approxPolyDP(contours[0], epsilon, True)
    cv2.imwrite("thresh.jpg", thresh)

    return transform_by4(image, approx[:, 0, :], width, height)

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

def crop_center(image, y, x, size):
    d = size // 2
    return image.crop((x - d, y - d, x + d + 1, y + d + 1))

def random_position(height, width, ratio):
    from random import randrange
    return randrange(height * ratio), randrange(width * ratio // 2)

def pick(y, x, arm, ratio):
    x //= ratio
    y //= ratio
    y = ARM_RANGE_HEIGHT - y
    arm.send_position(BASE_X + x, BASE_Y + y)
    print(BASE_X + x, BASE_Y + y)
    while True:
        res = arm.read_one_byte()
        print(res)
        if res != 0:
            return res == 11

def counter(res):
    result = []
    with open('day1.txt') as f:
        for line in f:
            result = [int(l) for l in line.split()]
    
    with open('day1.txt', 'w') as f:
        result[int(res)] += 1
        print(*result, file=f)

def add_red_point(pil_image, h, w):
    im = np.array(pil_image)
    for i in range(3):
        im[h][w][i] = 0
    im[h][w][0] = 255

    return Image.fromarray(im)


def main(model):
    INPUT_SIZE = 129
    BATCH = ARM_RANGE_WIDTH // 2
    OBJECT_NUM = 3
    picked_count = 0
    indicator = 0
    RATIO = 4  # the ratio of the arm position system to an image
    os.makedirs('entire', exist_ok=True)
    
    arm = armCommunication('COM8', 115200, 20)
    save_dirctory = './models/' + str(get_max_dir('./models') + 1)
    # os.makedirs(save_dirctory, exist_ok=True)
    net = AlexNet()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    net.to(device)
    if model is not None:
        net.load_state_dict(torch.load(model))
    net.eval()
    sigmoid = nn.Sigmoid()
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize(tuple([0.5] * 3), tuple([0.5] * 3))]
    )
    latest_positions = deque([(0, 0) for i in range(5)], maxlen=5)

    for i in range(int(1e6)):
        # if i != 0 and (i == 100 or i % 500 == 0):
        #     model_save_path = os.path.join(save_dirctory, '{}.pth'.format(i))
        #     train(os.path.join(model_save_path))
        #     net.load_state_dict(torch.load(model_save_path))
        #     net.eval()

        if picked_count >= OBJECT_NUM:
            picked_count = 0
            indicator = (indicator + 1) & 1

        print('cap')
        image = np_to_PIL(crop_image_along_line(capture(), ARM_RANGE_WIDTH * RATIO, ARM_RANGE_HEIGHT * RATIO))
        # image = Image.open('test/2539.jpg')
        print(image.size)
        print('done')

        P = np.zeros(shape=(ARM_RANGE_HEIGHT, ARM_RANGE_WIDTH), dtype=np.float16)

        with torch.no_grad():
            for h in range(ARM_RANGE_HEIGHT):
                input_images = []
                for w in range(ARM_RANGE_WIDTH // 2):
                    input_images.append(transform(crop_center(image, h * RATIO, (w + indicator * ARM_RANGE_WIDTH // 2) * RATIO, INPUT_SIZE)))
                
                outputs = sigmoid(net(torch.stack(input_images).to(device)))
                    
                for w in range(ARM_RANGE_WIDTH // 2):
                    P[h][w + indicator * ARM_RANGE_WIDTH // 2] = outputs[w]

        
        for i, (h, w) in enumerate(latest_positions, 1):
            for y in range(max(0, h - i ** 2), min(ARM_RANGE_HEIGHT, h + i ** 2 + 1)):
                for x in range(max(0, w - i ** 2), min(ARM_RANGE_WIDTH, w + i ** 2 + 1)):
                    P[y][x] = 0

        h, w = np.unravel_index(np.argmax(P), P.shape)
        print("probability:", P[h][w])

        overray = Image.fromarray(probability_to_green_image_array(P)).resize((ARM_RANGE_WIDTH * RATIO, ARM_RANGE_HEIGHT * RATIO))
        blended = add_red_point(Image.blend(image, overray, alpha=0.5), h * RATIO, w * RATIO)
        blended.show()

        latest_positions.append((h, w))
        h *= RATIO
        w *= RATIO
        time.sleep(1)  # what is this?
        try:
            res = pick(h, w, arm, RATIO)  # the position on the full image
        except Exception as e:
            print(e)
            continue
        picked_count += res
        image_save_path = './images/{}/{}.jpg'.format(int(res), get_max_file('./images/{}'.format(int(res))) + 1)
        crop_center(image, h, w, INPUT_SIZE).save(image_save_path)
        image.save('./entire/{}.jpg'.format(get_max_file('./entire') + 1))
        counter(res)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, default='70.pth')
    args = parser.parse_args()
    main(args.model)