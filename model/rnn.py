import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.utils as utils
import torch
import time
import math
from random import randint
import torch.utils.data.sampler as sampler
import torch.utils.data


class RNNModel(nn.Module):
    """
    input: (N * seq_len * feature_num)
    output: (N, 3)
    """
    def __init__(self, batch_size, rnn_len = 5, hidden_state=128, feature_num=29):
        super(RNNModel, self).__init__()
        self.n_layer = rnn_len
        self.nhid = hidden_state

        self.l0 = nn.Linear(feature_num, feature_num)

        # self.d1 = nn.Dropout(p=0.2)
        self.rnn = nn.LSTM(input_size=feature_num, hidden_size=hidden_state, num_layers=rnn_len, batch_first=True)
        # (N * 500 * 128)
        # (N * 128)
        self.l1 = nn.Linear(hidden_state, hidden_state)
        self.a1 = nn.Sigmoid()
        # (N * 128)

        # (N * 128)
        self.l2 = nn.Linear(hidden_state, 3)
        # (N * 3)
        self.softmax = nn.Softmax()

        # (100, 128)
        self.init_weights()

    def init_weights(self):
        self.l0.weight.data.uniform_(-.1, .1)
        self.l1.weight.data.uniform_(-.1, .1)
        self.l2.weight.data.uniform_(-.1, .1)
        self.l0.bias.data.fill_(0)
        self.l1.bias.data.fill_(0)
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
        o4 = self.a1(self.l1(o3))
        o5 = self.softmax(self.l2(o4))
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

def evaluate(model, loader):
    model.eval()
    d = next(iter(loader))
    hidden = model.init_hidden(len(d[0]))
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

def train(model, reader, exp_recorder, bsz=512, lr=0.005):
    model.train()
    total_loss = 0
    start_time = time.time()
    hidden = model.init_hidden(bsz=bsz)

    datas_arr = []
    targets_arr = []

    init_time = time.clock()

    total_loss = 0
    loader = torch.utils.data.DataLoader(dataset=reader, batch_size=bsz, shuffle=True)
    best_accu = 0
    for i, d in enumerate(loader):
        batch = i

        data = Variable(d[0].float())
        targets = Variable(target_onehot_to_classnum_tensor(d[1]))

        hidden = model.init_hidden(bsz)
        model.zero_grad()
        output, hidden = model(data, hidden)
        #print(output.size())
        #print(data)
        #print(output)
        #print(targets.size())
        # print(output, targets)
        loss = criterion(output, targets)
        loss.backward()

        # utils.clip_grad_norm(model.parameters(), 0.25)
        for p in model.parameters():
            p.data.add_(-lr, p.grad.data)

        total_loss += loss.data

        cur_loss = total_loss[0] / 1
        total_loss = 0
        elapsed = time.time() - start_time
        print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
              'loss {:5.2f} | ppl {:8.2f}'.format(
            0, batch, len(loader), 0.005,
            elapsed * 1000 / 1, cur_loss, math.exp(cur_loss)))
        start_time = time.time()

        exp_recorder.add_scalar_value("train_loss", cur_loss)

        test_loader = torch.utils.data.DataLoader(dataset=reader, batch_size=bsz, shuffle=True)
        test_loss, accu = evaluate(model, test_loader)
        test_loss = test_loss.data[0]
        print(test_loss, accu)

        if accu > best_accu:
            best_accu = accu

        exp_recorder.add_scalar_value('test_loss', test_loss, time.clock() - init_time)
        exp_recorder.add_scalar_value('test_accu', accu, time.clock() - init_time)
        datas_arr.clear()
        targets_arr.clear()