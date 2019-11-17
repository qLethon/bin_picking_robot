import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
import utils

import numpy as np
from PIL import Image

import Network
from Network import AlexNet
import utils
import settings

def make_train_set(model, images_path, save_path):

    os.makedirs(save_path, exist_ok=True)
    labels_path = os.path.join(save_path, "labels")
    blended_path = os.path.join(save_path, "blended")
    os.makedirs(labels_path, exist_ok=True)
    os.makedirs(blended_path, exist_ok=True)

    INPUT_SIZE = 129
    ARM_RANGE_WIDTH = settings.ARM_RANGE_WIDTH
    ARM_RANGE_HEIGHT = settings.ARM_RANGE_HEIGHT
    RATIO = settings.RATIO
    BATCH = ARM_RANGE_WIDTH

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
    images.sort(key=lambda path: int(os.path.basename(path).split('.')[0]))

    for image_path in images:
        image = Image.open(image_path)
        if image.size != (ARM_RANGE_WIDTH * RATIO, ARM_RANGE_HEIGHT * RATIO):
            image = image.resize((ARM_RANGE_WIDTH * RATIO, ARM_RANGE_HEIGHT * RATIO))
        save_path = os.path.join(labels_path, os.path.basename(image_path).split('.')[0])
        if os.path.exists(save_path):
            continue
        print(image_path)
        P = np.zeros((ARM_RANGE_HEIGHT, ARM_RANGE_WIDTH), dtype=np.float16)
        with torch.no_grad():
            for h in range(ARM_RANGE_HEIGHT):
                input_images = [transform(utils.crop_center(image, h * RATIO, w * RATIO, INPUT_SIZE)) for w in range(ARM_RANGE_WIDTH)]
                outputs = sigmoid(net(torch.stack(input_images).to(device)))
                for w, output in enumerate(outputs):
                    P[h][w] = output

        with open(save_path, 'w') as f:
            for p in P:
                for q in p:
                    print(q, end=' ', file=f)
                print(file=f)

        overray = Image.fromarray(utils.probability_to_green_image_array(P)).resize(image.size)
        blended = Image.blend(image, overray, alpha=0.5)
        blended.save(os.path.join(blended_path, os.path.basename(image_path)))

def valid_one(model_path, image_path):
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

def valid_fcnn(model_path, image_path, label_path=None):
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize(tuple([0.5] * 3), tuple([0.5] * 3))]
    )

    net = Network.FullyConvNet()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    net.to(device)
    net.load_state_dict(torch.load(model_path))
    net.eval()
    sigmoid = nn.Sigmoid()
    image = Image.open(image_path)
    with torch.no_grad():
        inputs = torch.stack([transform(image).to(device)])
        output = net(inputs)
    P = sigmoid(output).cpu().numpy()[0][0]

    overray = Image.fromarray(utils.probability_to_green_image_array(P))
    blended = Image.blend(image, overray, alpha=0.5)
    blended.show()
    blended.save('blended.jpg')

    if label_path is not None:
        with open(label_path) as fp:
            label = np.asarray([[float(s) for s in line.split()] for line in fp], dtype=np.float32)
        overray2 = Image.fromarray(utils.probability_to_green_image_array(label))
        blended = Image.blend(image, overray2, alpha=0.5)
        blended.show()

def valid(model_dir, test_dir, save_path):
    # train済みモデルから任意のテストデータでログ作るのを書こうとしたけどめんどくせ〜〜〜

    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize(tuple([0.5] * 3), tuple([0.5] * 3))]
    )
    BATCH = 32
    testset = torchvision.datasets.ImageFolder(test_dir, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH, shuffle=False, num_workers=2)
    net = AlexNet()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    net.eval()

    model_pathes = [f.path for f in os.scandir(model_dir) if f.is_file() and ".pth" in f.name]

    with torch.no_grad():
        for data in testloader:
            inputs, labels = data[0].to(device), data[1].view(-1, 1).float().to(device)
            outputs = net(inputs)
            val_loss += criterion(outputs, labels).item()
            results = sigmoid(outputs)
            total += labels.size(0)
            correct += (results.round() == labels).sum().item()

if __name__ == "__main__":
    valid_fcnn("test_fcnn_3/1090.pth", "FCNN/test/images/2562.jpg", "FCNN/test/labels/2562")