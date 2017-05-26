from reader.genetic import DBGeneticReader
import model.rnn as rnn
from pycrayon import CrayonClient
import datetime
from torch.utils.data.sampler import RandomSampler
import argparse
import torch


parser = argparse.ArgumentParser(description='RNN Future predictor')
parser.add_argument('--data', type=str, default='./processed/m0000.h5', help='location of DB file')
parser.add_argument('--cuda', action='store_true', help='whether use cuda')
parser.add_argument('--rec_url', type=str, default='orzserver.intmainreturn0.com', help='TensorBoard address')
parser.add_argument('--name', type=str, default='', help='Name of this experiment')
parser.add_argument('--read_first_k', type=int, default=-1, help='Only read first k table')
parser.add_argument('--read_old_model', type=str, default=None, help='Path of old model')
args = parser.parse_args()

with DBGeneticReader(args.data, read_first_k_table=args.read_first_k) as reader:
    if args.read_old_model is None:
        model = rnn.RNNModel(64, rnn_len=5)
        if args.cuda:
            model.cuda()
    else:
        model = torch.load(args.read_old_model)

    cc = CrayonClient(hostname=args.rec_url)
    exp = cc.create_experiment('Future_{}_{}'.format(args.name,
                                                     datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")))
    rnn.train(model, reader, exp, args, use_cuda=args.cuda)