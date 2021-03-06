import torch.nn as nn
from torch.autograd import Variable
from utils.index_list import SequentialIndexList
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
    output: (N, 2)
    """

    def __init__(self, batch_size, rnn_len=5, hidden_state=64, feature_num=29, var_hidden=None, dropout=False):
        super(RNNModel, self).__init__()
        self.n_layer = rnn_len
        self.nhid = hidden_state

        self.l0 = nn.Linear(feature_num, feature_num)

        # self.d1 = nn.Dropout(p=0.2)
        if var_hidden is None:
            self.rnn = nn.LSTM(input_size=feature_num, hidden_size=hidden_state, num_layers=rnn_len, batch_first=True)
            rnn_output_size = hidden_state
        else:
            self.hidden_arr = var_hidden
            for i, state_num in enumerate(var_hidden):
                assert (rnn_len == len(var_hidden))
                last_size = var_hidden[i - 1] if i > 0 else feature_num
                setattr(self, 'rnn_{}'.format(i),
                        nn.LSTM(input_size=last_size, hidden_size=state_num, num_layers=1, batch_first=True))
                rnn_output_size = var_hidden[-1]
        # (N * 500 * 128)
        # (N * 128)
        # self.l1 = nn.Linear(hidden_state, hidden_state)
        # self.a1 = nn.Sigmoid()
        # (N * 128)

        # (N * 128)
        self._dropout = dropout
        if dropout:
            self.do = nn.Dropout(p=0.2)

        self.l2 = nn.Linear(rnn_output_size, 2)
        # (N * 2)
        self.softmax = nn.Softmax()

        # (100, 128)
        self.init_weights()

    def init_weights(self):
        self.l0.weight.data.uniform_(-.1, .1)
        # self.l1.weight.data.uniform_(-.1, .1)
        self.l2.weight.data.uniform_(-.1, .1)
        self.l0.bias.data.fill_(0)
        # self.l1.bias.data.fill_(0)
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
        if hasattr(self, 'rnn'):
            o2, hidden2 = self.rnn(o1_x, hidden)  # N * 100 * 128
        elif hasattr(self, 'rnn_0'):
            hidden2 = [None for i in range(len(self.hidden_arr))]
            o2s = [o1_x]

            def closure(i, cell):
                nonlocal o2s, hidden2
                o2, hidden2[i] = cell(o2s[-1], hidden[i])
                o2s.append(o2)

            for i in range(len(self.hidden_arr)):
                cell = getattr(self, 'rnn_{}'.format(i), None)
                closure(i, cell)
            o2 = o2s[-1]

        o3 = o2[:, -1, :]  # N * 128
        if self._dropout:
            o4 = self.do(o3)
        else:
            o4 = o3
        # o4 = self.a1(self.l1(o3))
        o5 = self.softmax(self.l2(o4))
        return o5, hidden2

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if hasattr(self, 'rnn'):
            return (Variable(weight.new(self.n_layer, bsz, self.nhid).zero_()),
                    Variable(weight.new(self.n_layer, bsz, self.nhid).zero_()))
        else:
            res = []
            for i in range(len(self.hidden_arr)):
                res.append((Variable(weight.new(1, bsz, self.hidden_arr[i]).zero_()),
                            Variable(weight.new(1, bsz, self.hidden_arr[i]).zero_())))
            return res


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

    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(SequentialIndexList(0, int(len(reader) * .7)))
    loader = torch.utils.data.DataLoader(dataset=reader, batch_size=bsz, shuffle=True, sampler=train_sampler)

    test_sampler = torch.utils.data.sampler.SubsetRandomSampler(
        SequentialIndexList(int(len(reader) * .7), len(reader) - 1))
    test_loader = torch.utils.data.DataLoader(dataset=reader, batch_size=bsz, shuffle=True, sampler=test_sampler)

    best_loss = 100000

    optimizer = torch.optim.Adadelta(model.parameters(), weight_decay=0.001)
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
