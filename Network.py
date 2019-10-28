import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os


class AlexNet(nn.Module):
    #  for 129 x 129

    def __init__(self):
        super(AlexNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 96, 11, 3)
        self.pool1 = nn.MaxPool2d(3, 2)
        self.norm1 = nn.LocalResponseNorm(5, k=2)
        self.conv2 = nn.Conv2d(96, 256, 5, 1, padding=2)
        self.pool2 = nn.MaxPool2d(3, 2)
        self.norm2 = nn.LocalResponseNorm(5, k=1)
        self.conv3 = nn.Conv2d(256, 384, 3, 1, padding=1)
        self.conv4 = nn.Conv2d(384, 384, 3, 1, padding=1)
        self.conv5 = nn.Conv2d(384, 256, 3, 1, padding=1)
        self.pool5 = nn.MaxPool2d(3, 2)
        self.fc6 = nn.Linear(4096, 4096)
        self.fc7 = nn.Linear(4096, 1028)
        self.fc8 = nn.Linear(1028, 1)

    def forward(self, x):
        x = self.norm1(self.pool1(F.relu(self.conv1(x))))
        x = self.norm2(self.pool2(F.relu(self.conv2(x))))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool5(F.relu(self.conv5(x)))
        x = x.view(-1, 4096)
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = self.fc8(x)
        
        return x


class AlexNet2(nn.Module):
    #  for 65 x 65

    def __init__(self):
        super(AlexNet2, self).__init__()

        self.conv1 = nn.Conv2d(3, 96, 11, 2)
        self.pool1 = nn.MaxPool2d(3, 2)
        self.norm1 = nn.LocalResponseNorm(5, k=2)
        self.conv2 = nn.Conv2d(96, 256, 5, 1, padding=2)
        self.pool2 = nn.MaxPool2d(3, 2)
        self.norm2 = nn.LocalResponseNorm(5, k=1)
        self.conv3 = nn.Conv2d(256, 384, 3, 1, padding=1)
        self.conv4 = nn.Conv2d(384, 384, 3, 1, padding=1)
        self.conv5 = nn.Conv2d(384, 256, 3, 1, padding=1)
        self.pool5 = nn.MaxPool2d(3, 2)
        self.fc6 = nn.Linear(1024, 1024)
        self.fc7 = nn.Linear(1024, 256)
        self.fc8 = nn.Linear(256, 1)

    def forward(self, x):
        x = self.norm1(self.pool1(F.relu(self.conv1(x))))
        x = self.norm2(self.pool2(F.relu(self.conv2(x))))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool5(F.relu(self.conv5(x)))
        x = x.view(-1, 1024)
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = self.fc8(x)
        
        return x

class Net(nn.Module):
    #  for 64 x 64

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5, 2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(400, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 400)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train(save_path):

    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize(tuple([0.5] * 3), tuple([0.5] * 3))]
    )
    trainset = torchvision.datasets.ImageFolder('./images', transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=8, shuffle=True, num_workers=2)

    # testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

    net = AlexNet()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cuda")
    # net.cuda()
    net.to(device)

    invalid_num = len([path for path in os.listdir('./images/0') if os.path.isfile(os.path.join('./images/0', path))])
    valid_num = len([path for path in os.listdir('./images/1') if os.path.isfile(os.path.join('./images/1', path))])
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([invalid_num / valid_num]).to(device))
    # criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


    for epoch in range(25):
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            inputs, labels = data[0].to(device), data[1].view(-1, 1).float().to(device)
            optimizer.zero_grad()

            outputs = net(inputs)
            m = nn.Softmax()
            print(outputs, labels)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 1999:
                print('[%d, %5d] loss: %.3f' % 
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
        print('[%d] loss: %.3f' % 
            (epoch + 1, running_loss / (invalid_num + valid_num)))

    print('Finished Training')
    torch.save(net.state_dict(), save_path)


import multiprocessing

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', True)
    train('test.pth')