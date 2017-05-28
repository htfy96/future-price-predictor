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

class cnnT1(nn.Module):

    def __init__(self):
        super(cnnT1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(16 * 25 * 7, 120)
        self.fc2 = nn.Linear(120, 3)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.softmax(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

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
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
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
                torch.save(model.state_dict(), os.path.join('saved_model', 'cnnT1_epoch_{}_i_{}_acc_{}.t7'.format(epoch + 1, i, acc)))
    acc = evaluate(model, trainloader, use_cuda)
    torch.save(model.state_dict(), os.path.join('saved_model', 'cnnT1_all_acc_{}.t7'.format(acc)))

    print("Finished Training!")


