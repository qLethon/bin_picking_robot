import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
import argparse

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
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.norm1(self.pool1(F.relu(self.conv1(x))))
        x = self.norm2(self.pool2(F.relu(self.conv2(x))))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool5(F.relu(self.conv5(x)))
        x = x.view(-1, 4096)
        x = self.dropout(F.relu(self.fc6(x)))
        x = self.dropout(F.relu(self.fc7(x)))
        x = self.fc8(x)
        
        return x

class ValNet(AlexNet):

    def _forward(self, x):
        x = self.norm1(self.pool1(F.relu(self.conv1(x))))
        x = self.norm2(self.pool2(F.relu(self.conv2(x))))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool5(F.relu(self.conv5(x)))
        x = x.view(-1, 4096)
        x = self.dropout(F.relu(self.fc6(x)))
        x = self.dropout(F.relu(self.fc7(x)))
        x = self.fc8(x)

    def forward(self, x):
        res = [list() for i in range(256)]
        print(x.size())
        dh = x.size()[2] // 256
        dw = x.size()[3] // 256
        INPUT_SIZE = 129
        for h in range(255):
            for w in range(255):
                    res[h].append(self._forward(x[:, :, h * dh:h * dh + INPUT_SIZE , w * dw:w * dw + INPUT_SIZE]))

        return torch.stack(res)

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
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.norm1(self.pool1(F.relu(self.conv1(x))))
        x = self.norm2(self.pool2(F.relu(self.conv2(x))))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool5(F.relu(self.conv5(x)))
        x = x.view(-1, 1024)
        x = self.dropout(F.relu(self.fc6(x)))
        x = self.dropout(F.relu(self.fc7(x)))
        x = self.fc8(x)
        
        return x

class FullyConvNet(nn.Module):

    def __init__(self):
        super(FullyConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 9, padding=4)
        self.conv2 = nn.Conv2d(16, 16, 5, padding=2)
        # max_pool
        self.conv3 = nn.Conv2d(16, 16, 5, padding=2)
        self.conv4 = nn.Conv2d(16, 16, 5, padding=2)
        # max_pool
        self.conv5 = nn.Conv2d(16, 128, 19, padding=9)
        self.conv6 = nn.Conv2d(128, 128, 1)
        self.conv7 = nn.Conv2d(128, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)

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
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 400)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


def train(save_dir):
    BATCH = 32

    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize(tuple([0.5] * 3), tuple([0.5] * 3))]
    )
    trainset = torchvision.datasets.ImageFolder('./images', transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH, shuffle=True, num_workers=2)

    testset = torchvision.datasets.ImageFolder('./data', transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH, shuffle=False, num_workers=2)

    net = AlexNet()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cuda")
    # net.cuda()
    net.to(device)

    invalid_num = len([path for path in os.listdir('./images/0') if os.path.isfile(os.path.join('./images/0', path))])
    valid_num = len([path for path in os.listdir('./images/1') if os.path.isfile(os.path.join('./images/1', path))])
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([invalid_num / valid_num]).to(device))
    optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)
    sigmoid = nn.Sigmoid()

    for epoch in range(5000):
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            inputs, labels = data[0].to(device), data[1].view(-1, 1).float().to(device)
            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        train_loss = running_loss / ((invalid_num + valid_num + BATCH - 1) // BATCH)
        print('[{}] loss: {}'.format(epoch + 1, train_loss))
        with open(os.path.join(save_dir, "loss.log"), 'a') as f:
            print(epoch + 1, train_loss, file=f)
        if epoch % 10 == 0:
            torch.save(net.state_dict(), os.path.join(save_dir, "{}.pth".format(epoch)))
            correct = 0
            total = 0
            val_loss = 0
            net.eval()
            with torch.no_grad():
                for data in testloader:
                    inputs, labels = data[0].to(device), data[1].view(-1, 1).float().to(device)
                    outputs = net(inputs)
                    val_loss += criterion(outputs, labels).item()
                    results = sigmoid(outputs)
                    total += labels.size(0)
                    correct += (results.round() == labels).sum().item()
            net.train()

            val_loss /= (len(testset) + BATCH - 1) // BATCH
            acc = correct / total
            print('[{}] acc: {}'.format(epoch + 1, acc))
            print('[{}] val: {}'.format(epoch + 1, val_loss))
            with open(os.path.join(save_dir, "acc.log"), 'a') as f:
                print(epoch + 1, acc, file=f)
            with open(os.path.join(save_dir, "val.log"), 'a') as f:
                print(epoch + 1, val_loss, file=f)

    print('Finished Training, save to' + save_dir)
    torch.save(net.state_dict(), os.path.join(save_dir, "final.pth"))

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method('spawn', True)
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--savedir', type=str, required=True)
    args = parser.parse_args()
    os.makedirs(args.savedir, exist_ok=True)
    train(args.savedir)