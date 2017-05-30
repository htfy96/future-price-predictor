import torch
from torch.autograd import Variable
import torch.utils.data as data_utils
import argparse
import datetime
import os.path
import re
from pycrayon import CrayonClient
import reader.genetic as DBGR
import model.cnnT2 as drn
from utils.index_list import SequentialIndexList

cuda_ava = torch.cuda.is_available()

classes = ["drop", "hold", "rise"]

modelPath = "./saved_model/"

def vec_to_classnum(onehot):
    return torch.max(onehot, -1)[1][0]

def target_onehot_to_classnum_tensor(target_onehot_vec):
    return torch.LongTensor([vec_to_classnum(x) for x in target_onehot_vec])

def evaluate(model, testloader, args, use_cuda=False):
    correct = 0
    total = 0
    class_correct = list(0. for i in range(3))
    class_total = list(0. for i in range(3))
    for i, data in enumerate(testloader, 0):
        if i == 100:
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
        c = (predicted == targets).squeeze()
        for i in range(args.batch_size):
            target = targets[i]
            class_correct[target] += c[i]
            class_total[target] += 1

    print("Accuracy of the network is: %.5f %%" % (correct / total * 100))

    for i in range(3):
        if class_total[i] == 0:
            print("Accuracy of %1s : %1s %% (%1d / %1d)" % (classes[i], "NaN", class_correct[i], class_total[i]))
        else:
            print("Accuracy of %1s : %.5f %% (%1d / %1d)" % (classes[i], class_correct[i] / class_total[i] * 100, class_correct[i], class_total[i]))

    return correct / total

parser = argparse.ArgumentParser(description='CNN Future predictor')
parser.add_argument('--data', type=str, default='./processed/m0000.h5', help='location of DB file')
parser.add_argument('--cuda', action='store_true', default=True, help='whether use cuda')
parser.add_argument('--rec_url', type=str, default='orzserver.intmainreturn0.com', help='TensorBoard address')
parser.add_argument('--read_first_k', type=int, default=-1, help='Only read first k table')
parser.add_argument('--test', action='store_true', default=True, help='Rest existing models')
parser.add_argument('--name', type=str, default='cnnT2', help='Name of this predictor')
parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--epoch', type=int, default=1, help='epoch number')
args = parser.parse_args()

with DBGR.DBGeneticReader(db_h5_file=args.data, read_first_k_table=args.read_first_k) as db:
    if cuda_ava:
        if args.cuda:
            with torch.cuda.device(args.gpu):
                model = drn.cnnT2(drn.ResidualBlock, [2, 2, 2, 2])
                model = model.cuda()
                if args.test is False:
                    cc = CrayonClient(hostname=args.rec_url)
                    exp = cc.create_experiment(
                        '{}_{}'.format(args.name, datetime.datetime.now().strftime("%b_%d_%H:%M:%S")))
                    drn.train(model, db, exp, args, use_cuda=args.cuda)
                else:
                    #testsampler = data_utils.sampler.SubsetRandomSampler(SequentialIndexList(int(len(db) * 0.7), len(db) - 1))
                    #testloader = data_utils.DataLoader(dataset=db, batch_size=args.batch_size, shuffle=True, sampler=testsampler)
                    testloader = data_utils.DataLoader(dataset=db, batch_size=args.batch_size, shuffle=True)
                    for parent, dirnames, filenames in os.walk(modelPath):
                        for filename in filenames:
                            str = re.split("_|\.", filename)
                            if str[0] == args.name and str[3] == "iter" and len(str[4]) == 4 and str[4] >= "2000":
                                model.load_state_dict(torch.load(os.path.join(parent, filename)))
                                print(filename)
                                evaluate(model, testloader, args, use_cuda=True)
        else:
            print("Please use GPU to train/test this network!")
    else:
        print("Sorry, GPU is not available!")