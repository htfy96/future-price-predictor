import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils
import numpy as np
import os
import reader.genetic as DBGR

cuda_ava = torch.cuda.is_available()

def vec_to_classnum(onehot):
    return torch.max(onehot, -1)[1][0]

def target_onehot_to_classnum_tensor(target_onehot_vec):
    return torch.LongTensor([vec_to_classnum(x) for x in target_onehot_vec])

# 3x3 Convolution
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)


# Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


# ResNet Module
class cnnT2(nn.Module):
    def __init__(self, block, layers, num_classes=3):
        super(cnnT2, self).__init__()
        self.in_channels = 16
        self.conv = conv3x3(1, 16)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 16, layers[0])
        self.layer2 = self.make_layer(block, 32, layers[0])
        self.layer3 = self.make_layer(block, 64, layers[1])
        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(2304, num_classes)
        self.softmax = nn.Softmax()

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        print(out.size())
        out = self.fc(out)
        out = self.relu(out)
        print(out.size())
        out = self.softmax(out)
        return out


def evaluate(model, testloader, use_cuda=False):
    correct = 0
    total = 0
    for i, data in enumerate(testloader, 0):
        if i == 10:
            break
        inputs, targets = data
        inputs = inputs.unsqueeze(1)
        targets = target_onehot_to_classnum_tensor(targets)
        if use_cuda and cuda_ava:
            inputs = Variable(inputs.float().cuda())
            targets = targets.cuda()
        else:
            inputs = Variable(inputs.float())
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum()

    print("Accuracy of the network is: %.5f %%" % (correct / total * 100))
    return correct / total

def train(model, db, args, bsz=32, eph=1, use_cuda=False):
    print("Training...")

    trainloader = data_utils.DataLoader(dataset=db, batch_size=bsz, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    best_loss = 100000

    for epoch in range(eph):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 1):
            inputs, targets = data
            inputs = inputs.unsqueeze(1)
            targets = target_onehot_to_classnum_tensor(targets)
            if use_cuda and cuda_ava:
                inputs = Variable(inputs.float().cuda())
                targets = Variable(targets.cuda())
            else:
                inputs = Variable(inputs.float())
                targets = Variable(targets)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.data[0]
            last_loss = loss.data[0]
            if i % 100 == 0:
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i, running_loss / 100))
                running_loss = 0

            if last_loss < best_loss:
                best_loss = last_loss
                acc = evaluate(model, trainloader, use_cuda)
                torch.save(model.state_dict(), os.path.join('saved_model', 'cnnT2_epoch_{}_i_{}_acc_{}.t7'.format(epoch + 1, i, acc)))
    acc = evaluate(model, trainloader, use_cuda)
    torch.save(model.state_dict(), os.path.join('saved_model', 'cnnT2_all_acc_{}.t7'.format(acc)))

    print("Finished Training!")