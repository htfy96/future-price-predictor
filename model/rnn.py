import torch.nn as nn
from torch.autograd import Variable
import torch
import time
import math
import os
import torch.utils.data.sampler as sampler
import torch.utils.data
import torch.optim
import datetime


class RNNModel(nn.Module):
    """
    input: (N * seq_len * feature_num)
    output: (N, 3)
    """
    def __init__(self, batch_size, rnn_len = 5, hidden_state=64, feature_num=29):
        super(RNNModel, self).__init__()
        self.n_layer = rnn_len
        self.nhid = hidden_state

        self.l0 = nn.Linear(feature_num, feature_num)

        # self.d1 = nn.Dropout(p=0.2)
        self.rnn = nn.LSTM(input_size=feature_num, hidden_size=hidden_state, num_layers=rnn_len, batch_first=True)
        # (N * 500 * 128)
        # (N * 128)
        #self.l1 = nn.Linear(hidden_state, hidden_state)
        # self.a1 = nn.Sigmoid()
        # (N * 128)

        # (N * 128)
        self.l2 = nn.Linear(hidden_state, 3)
        # (N * 3)
        self.softmax = nn.Softmax()

        # (100, 128)
        self.init_weights()

    def init_weights(self):
        self.l0.weight.data.uniform_(-.1, .1)
        #self.l1.weight.data.uniform_(-.1, .1)
        self.l2.weight.data.uniform_(-.1, .1)
        self.l0.bias.data.fill_(0)
        #self.l1.bias.data.fill_(0)
        self.l2.bias.data.fill_(0)

    def forward(self, input, hidden):
        """
        :param input: (N, seq, input_feature)
        :param hidden: ((n_layer, batch_size, hidden_feature), ---_
        :return: (output, hidden_out)
        """
        input_sz = input.size()
        ii = input.resize(input_sz[0] * input_sz[1], input_sz[2])
        o1 = self.l0(ii)
        o1_x = o1.resize(input_sz[0], input_sz[1], input_sz[2])
        o2, hidden2 = self.rnn(o1_x, hidden) # N * 100 * 128
        o3 = o2[:, -1, :] # N * 128
        # o4 = self.l1(o3)
        o5 = self.softmax(self.l2(o3))
        return o5, hidden2

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (Variable(weight.new(self.n_layer, bsz, self.nhid).zero_()),
                Variable(weight.new(self.n_layer, bsz, self.nhid).zero_()))

def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)

criterion = nn.CrossEntropyLoss()

def evaluate(model, loader, use_cuda=False):
    model.eval()
    d = next(iter(loader))
    hidden = model.init_hidden(len(d[0]))
    if use_cuda:
        data_var = Variable(d[0].float().cuda(), volatile=True)
        target_var = Variable(target_onehot_to_classnum_tensor(d[1]).cuda(), volatile=True)
    else:
        data_var = Variable(d[0].float(), volatile=True)
        target_var = Variable(target_onehot_to_classnum_tensor(d[1]), volatile=True)
    output, hidden = model(data_var, hidden)
    return criterion(output, target_var), get_accu(output, d[1])

def vec_to_classnum(onehot):
    return torch.max(onehot, -1)[1][0]

def target_onehot_to_classnum_tensor(target_onehot_vec):
    return torch.LongTensor([vec_to_classnum(x) for x in target_onehot_vec])

def get_accu(output_vec, target_vec):
    output_class_vec = [vec_to_classnum(x) for x in output_vec.data]
    total_accu = 0.0
    for i, out_class in enumerate(output_class_vec):
        if target_vec[i][out_class] == 1:
            total_accu += 1.0
    return total_accu / len(output_vec)

def train(model, reader, exp_recorder, args, bsz=512, use_cuda=False):
    start_time = time.time()

    hidden = model.init_hidden(bsz=bsz)

    datas_arr = []
    targets_arr = []

    init_time = time.clock()

    loader = torch.utils.data.DataLoader(dataset=reader, batch_size=bsz, shuffle=True)
    best_accu = 0

    best_loss = 100000

    optimizer = torch.optim.Adadelta(model.parameters())
    for i, d in enumerate(loader):
        model.train()
        batch = i

        if use_cuda:
            data = Variable(d[0].float().cuda())
            targets = Variable(target_onehot_to_classnum_tensor(d[1]).cuda())
        else:
            data = Variable(d[0].float())
            targets = Variable(target_onehot_to_classnum_tensor(d[1]))
        last_loss = 100000

        def closure():
            optimizer.zero_grad()
            hid = model.init_hidden(bsz)
            out, hid = model(data, hid)
            loss2 = criterion(out, targets)
            loss2.backward()
            nonlocal start_time
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                0, batch, len(loader), 0.005,
                elapsed * 1000 / 1, loss2.data[0], math.exp(loss2.data[0])))
            exp_recorder.add_scalar_value("train_loss", loss2.data[0], time.clock() - init_time)
            start_time = time.time()
            nonlocal  last_loss
            last_loss = loss2.data[0]
            return loss2
        optimizer.step(closure)

        if last_loss < best_loss or batch % 10 == 0:
            best_loss = last_loss
            test_loader = torch.utils.data.DataLoader(dataset=reader, batch_size=bsz, shuffle=True)
            test_loss, accu = evaluate(model, test_loader, use_cuda=use_cuda)
            test_loss = test_loss.data[0]
            print(test_loss, accu)
            exp_recorder.add_scalar_value('test_loss', test_loss, time.clock() - init_time)
            exp_recorder.add_scalar_value('test_accu', accu, time.clock() - init_time)
            torch.save(model,
                       os.path.join('saved_model',
                                    '{}_loss_test_{}_accu_{}_{}.t7'.format(
                                        args.name, test_loss, accu,
                                        datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")
                                    )))

        datas_arr.clear()
        targets_arr.clear()