import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms
import os
from Network import AlexNet
from Network import train
import Network
from PIL import Image
import numpy as np
import argparse
from interfaces import pick
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

def main(model):
    INPUT_SIZE = 129
    BATCH = 256
    save_dirctory = './models/' + str(get_max_dir('./models') + 1)
    os.makedirs(save_dirctory, exist_ok=True)
    net = Network.AlexNet()
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

        image = Image.open('./images/1/1.jpg')  # TODO: via webcam
        # image = capture(2)
        # TODO: crop and rotate an image alona a red rectangle

        dh = image.height // 256
        dw = image.width // 256
        # P = np.ndarray(shape=(image.height, image.width), dtype=float)
        x = to_tensor(image).to(device)
        P = []

        with torch.no_grad():
            for h in range(256):
                if (h * dh + INPUT_SIZE >= image.height):
                    break
                input_images = []
                for w in range(256):
                    if (w * dw + INPUT_SIZE >= image.width):
                        break
                    input_images.append(x[:, h * dh:h * dh + INPUT_SIZE , w * dw:w * dw + INPUT_SIZE])
                outputs = net(torch.stack(input_images))
                print(h)
                P.append(outputs)
            
        print(P)
        # max_h, max_w = np.unravel_index(np.argmax(P), P.shape)
        # print(np.unravel_index(np.argmax(P), P.shape))
        try:
            res = pick(max_w * 255 // image.width, max_h * 255 // image.height)
        except Exception as e:
            print(e)
            continue
        res = 1
        image_save_path = './images/{}/{}.jpg'.format(res, get_max_file('./images/{}'.format(res)) + 1)
        crop_center(image, max_w, max_h, INPUT_SIZE).save(image_save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str)
    args = parser.parse_args()
    main(args.model)