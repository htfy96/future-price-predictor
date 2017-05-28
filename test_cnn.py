import reader.genetic as DBGR
import model.cnnT1 as cnn
#from pycrayon import CrayonClient
import datetime
from torch.utils.data.sampler import RandomSampler
import argparse
import torch
from torch.autograd import Variable
import torch.utils.data as data_utils
import os
import os.path
import re

cuda_ava = torch.cuda.is_available()

modelPath = "./saved_model/"
model = cnn.cnnT1()
if cuda_ava:
    model = model.cuda()

def vec_to_classnum(onehot):
    return torch.max(onehot, -1)[1][0]

def target_onehot_to_classnum_tensor(target_onehot_vec):
    return torch.LongTensor([vec_to_classnum(x) for x in target_onehot_vec])

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
        #print(outputs)
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum()

    print("Accuracy of the network is: %.5f %%" % (correct / total * 100))
    return correct / total

with DBGR.DBGeneticReader('./processed/m0000.h5', read_first_k_table=1) as db:
    testloader = data_utils.DataLoader(dataset=db, batch_size=32, shuffle=True)
    for parent, dirnames, filenames in os.walk(modelPath):
        for filename in filenames:
            str = re.split("_|\.", filename)
            if str[0] == "cnnT1":
                model.load_state_dict(torch.load(os.path.join(parent, filename)))
                print(filename)
                evaluate(model, testloader, use_cuda=True)