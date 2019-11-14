import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
import argparse
import utils
from Network import FullyConvNet
from PIL import Image
import numpy as np

ARM_RANGE_HEIGHT = 87
ARM_RANGE_WIDTH = 250

class Dataset(torch.utils.data.Dataset):

    def __init__(self, image_dir, label_dir, transform=transforms.ToTensor()):
        to_tensor = transforms.ToTensor()

        images = set(f.name.rstrip('.jpg') for f in os.scandir(image_dir) if f.is_file())
        labels = set(f.name for f in os.scandir(label_dir) if f.is_file())
        datasets = tuple(images & labels)
        self.images = [transform(Image.open(os.path.join(image_dir, f + ".jpg")).resize((ARM_RANGE_WIDTH, ARM_RANGE_HEIGHT))) for f in datasets]
        self.labels = [to_tensor(self._get_label(os.path.join(label_dir, f))) for f in datasets]
     
    def __len__(self):
        return len(self.images)

    def _get_label(self, path):
        with open(path) as fp:
            label = np.asarray([[float(s) for s in line.split()] for line in fp], dtype=np.float32)
        ma = label.max()
        if label.shape != (ARM_RANGE_HEIGHT, ARM_RANGE_WIDTH):
            label = Image.fromarray((label * 255).astype(np.uint8)).resize((ARM_RANGE_WIDTH, ARM_RANGE_HEIGHT))
            label = np.asarray(label, dtype=np.float32) / 255
        return label

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

def train(save_dir, train_dir, test_dir):
    os.makedirs(save_dir, exist_ok=True)
    BATCH = 32

    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize(tuple([0.5] * 3), tuple([0.5] * 3))]
    )

    trainset = Dataset(os.path.join(train_dir, "images"), os.path.join(train_dir, "labels"), transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH, shuffle=True, num_workers=2)

    testset = Dataset(os.path.join(test_dir, "images"), os.path.join(test_dir, "labels"), transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH, shuffle=False, num_workers=2)

    net = FullyConvNet()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    print(device)
    print(len(trainset), len(trainloader))

    criterion = nn.L1Loss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    sigmoid = nn.Sigmoid()

    for epoch in range(5000):
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()

            outputs = sigmoid(net(inputs))
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        train_loss = running_loss / len(trainloader)
        print('[{}] loss: {}'.format(epoch + 1, train_loss))
        with open(os.path.join(save_dir, "loss.log"), 'a') as f:
            print(epoch + 1, train_loss, file=f)
        if epoch % 10 == 0:
            torch.save(net.state_dict(), os.path.join(save_dir, "{}.pth".format(epoch)))
            val_loss = 0
            net.eval()
            with torch.no_grad():
                for data in testloader:
                    inputs, labels = data[0].to(device), data[1].to(device)
                    outputs = sigmoid(net(inputs))
                    val_loss += criterion(outputs, labels).item()
            net.train()

            val_loss /= len(testloader)
            print('[{}] val: {}'.format(epoch + 1, val_loss))
            with open(os.path.join(save_dir, "val.log"), 'a') as f:
                print(epoch + 1, val_loss, file=f)

    print('Finished Training, save to' + save_dir)
    torch.save(net.state_dict(), os.path.join(save_dir, "final.pth"))

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method('spawn', True)
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--savedir', type=str)
    parser.add_argument('-t', '--traindir', type=str)
    parser.add_argument('-v', '--testdir', type=str)
    args = parser.parse_args()
    train(args.savedir, args.traindir, args.testdir)
    # train("test_fcnn_2", "FCNN/train/", "FCNN/test/")