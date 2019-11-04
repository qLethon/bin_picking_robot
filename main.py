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

def capture(num):
    cam = cv2.VideoCapture(num)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    # cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 4000)
    retval, frame = cam.read()
    if not retval:
        print('cannnot read')
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

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

def pick(y, x, indicator):
    base_x = -120
    base_y = 75
    half_x_point = 135
    #serial return pick(indicator * half_x_point + base_x + x, base_y + y)


def main(model):
    INPUT_SIZE = 129
    BATCH = 256
    OBJECT_NUM = 10
    picked_count = 0
    indicator = 0
    
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

        image = capture(2)
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
            res = pick(h, w, indicator)
        except Exception as e:
            print(e)
            continue
        picked_count += res
        image_save_path = './images/{}/{}.jpg'.format(res, get_max_file('./images/{}'.format(res)) + 1)
        crop_center(image, w, h, INPUT_SIZE).save(image_save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str)
    args = parser.parse_args()
    main(args.model)