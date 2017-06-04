import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils
import os.path
import datetime
import time
from utils.index_list import SequentialIndexList

cuda_ava = torch.cuda.is_available()

classes = ["drop", "rise"]

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

'''
# cnnB3
class cnnBeta(nn.Module):
    def __init__(self, block, layers, num_classes=2):
        super(cnnBeta, self).__init__()
        self.in_channels = 8
        self.conv = conv3x3(1, 8)
        self.bn = nn.BatchNorm2d(8)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 16, layers[0])
        self.layer2 = self.make_layer(block, 32, layers[1])
        self.layer3 = self.make_layer(block, 64, layers[2])
        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(1152, num_classes)
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
        # out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = self.relu(out)
        out = self.softmax(out)
        return out
'''

# ResNet Module
class cnnBeta(nn.Module):
    def __init__(self, block, layers, num_classes=2):
        super(cnnBeta, self).__init__()
        self.in_channels = 8
        self.conv = conv3x3(1, 8)
        self.bn = nn.BatchNorm2d(8)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 16, layers[0])
        self.layer2 = self.make_layer(block, 32, layers[1])
        self.layer3 = self.make_layer(block, 16, layers[2])
        self.conv2 = conv3x3(16, 8)
        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(288, num_classes)
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
        out = self.conv2(out)
        out = self.bn(out)
        out = self.relu(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = self.relu(out)
        out = self.softmax(out)
        return out


def evaluate(model, testloader, args, use_cuda=False):
    correct = 0
    total = 0
    class_correct = list(0. for i in range(2))
    class_total = list(0. for i in range(2))
    for i, data in enumerate(testloader, 0):
        if i == 20:
            break;
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
        c = (predicted == targets).squeeze()
        for i in range(args.batch_size):
            target = targets[i]
            class_correct[target] += c[i]
            class_total[target] += 1

    print("Accuracy of the network is: %.5f %%" % (correct / total * 100))

    for i in range(2):
        if class_total[i] == 0:
            print("Accuracy of %1s : %1s %% (%1d / %1d)" % (classes[i], "NaN", class_correct[i], class_total[i]))
        else:
            print("Accuracy of %1s : %.5f %% (%1d / %1d)" % (classes[i], class_correct[i] / class_total[i] * 100, class_correct[i], class_total[i]))

    return correct / total

def train(model, db, exp, args, use_cuda=False):
    print("Training...")
    init_time = time.clock()

    trainsampler = data_utils.sampler.SubsetRandomSampler(SequentialIndexList(0, int(len(db) * 0.7)))
    trainloader = data_utils.DataLoader(dataset=db, batch_size=args.batch_size, shuffle=True, sampler=trainsampler)
    testsampler = data_utils.sampler.SubsetRandomSampler(SequentialIndexList(int(len(db) * 0.7), len(db) - 1))
    testloader = data_utils.DataLoader(dataset=db, batch_size=args.batch_size, shuffle=True, sampler=testsampler)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    best_loss = 100000

    for epoch in range(args.epoch):
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

            if last_loss < best_loss or i % 100 == 0:
                best_loss = last_loss
                acc = evaluate(model, testloader, args, use_cuda)
                exp.add_scalar_value('train_loss', last_loss, time.clock() - init_time)
                exp.add_scalar_value('test_accu', acc, time.clock() - init_time)
                torch.save(model.state_dict(), os.path.join('saved_model', '{}_epoch_{}_iter_{}_loss_{}_acc_{}_{}.t7'.format(args.name, epoch + 1, i, last_loss, acc, datetime.datetime.now().strftime("%b_%d_%H:%M:%S"))))
    acc = evaluate(model, testloader, args, use_cuda)
    torch.save(model.state_dict(), os.path.join('saved_model', '{}_all_acc_{}.t7'.format(args.name, acc)))

    print("Finished Training!")