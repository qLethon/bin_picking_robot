import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os

import numpy as np
from PIL import Image

import Network
from Network import AlexNet


def probability_to_green_image_array(P):
    import numpy as np
    P *= 255
    green = np.zeros((P.shape[0], P.shape[1], 3), dtype=np.uint8)
    for h in range(P.shape[0]):
        for w in range(P.shape[1]):
            green[h][w][1] = P[h][w]

    return green

def probability_to_green_image_array(P, h, w):
    import numpy as np
    green = probability_to_green_image_array(P)
    P[h][w][2] = 255

    return green

def make_train_set(model, images_path):
    from main import crop_center

    basic_path = "dataset"
    os.makedirs("dataset", exist_ok=True)
    INPUT_SIZE = 129
    BATCH = 500

    net = AlexNet()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    net.to(device)
    net.load_state_dict(torch.load(model))
    net.eval()
    sigmoid = nn.Sigmoid()
    
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize(tuple([0.5] * 3), tuple([0.5] * 3))]
    )

    images = [f.path for f in os.scandir(images_path) if f.is_file()]

    for image_path in images:
        image = Image.open(image_path)
        # image_tensor = to_tensor(image).to(device)
        # print(image_tensor.size())
        P = np.zeros((image.height, image.width), dtype=np.float16)
        # P = np.random.ze(image.height, image.width)
        with torch.no_grad():
            for h in range(image.height):
                for w in range(0, image.width, BATCH):
                    # input_images = [image_tensor[:, max(0, h - INPUT_SIZE // 2):h + INPUT_SIZE // 2 + 1, rw - 129 // 2:rw + INPUT_SIZE // 2 + 1] for rw in range(w, w + BATCH)]
                    input_images = [transform(crop_center(image, h, rw, INPUT_SIZE)).to(device) for rw in range(w, w + BATCH)]
                    outputs = sigmoid(net(torch.stack(input_images)))
                    for i, output in enumerate(outputs):
                        P[h][w + i] = output

            save_path = os.path.join(basic_path, os.path.basename(image_path).split('.')[0])
            with open(save_path, 'w') as f:
                for p in P:
                    for q in p:
                        print(q, end=' ', file=f)
                    print(file=f)
            # overray = Image.fromarray(probability_to_green_image_array(P))
            # blended = Image.blend(image, overray, alpha=0.5)
            # blended.show()
            # blended.save('belended.jpg')

def valid(model_path, image_path):
    basic_path = "dataset"
    os.makedirs("dataset", exist_ok=True)
    INPUT_SIZE = 129
    BATCH = 250
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize(tuple([0.5] * 3), tuple([0.5] * 3))]
    )

    net = AlexNet()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    net.to(device)
    net.load_state_dict(torch.load(model_path))
    net.eval()
    sigmoid = nn.Sigmoid()
    with torch.no_grad():
        inputs = torch.stack([transform(Image.open(image_path)).to(device)])
        output = net(inputs)
    return sigmoid(output)