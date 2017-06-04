from reader.genetic import DBGeneticReader
import model.rnn as rnn
from pycrayon import CrayonClient
import datetime
from torch.utils.data.sampler import RandomSampler
import argparse
import torch

parser = argparse.ArgumentParser(description='RNN Future predictor')
parser.add_argument('--data', type=str, default='./processed/v2-m0000.h5', help='location of DB file')
parser.add_argument('--cuda', action='store_true', help='whether use cuda')
parser.add_argument('--rec_url', type=str, default='orzserver.intmainreturn0.com', help='TensorBoard address')
parser.add_argument('--name', type=str, default='', help='Name of this experiment')
parser.add_argument('--read_first_k', type=int, default=-1, help='Only read first k table')
parser.add_argument('--read_old_model', type=str, default=None, help='Path of old model')
parser.add_argument('--gpu', type=int, default=3, help='Number of gpu')
parser.add_argument('--normalize', type=bool, default=False, help='Whether to perform per-table normalize')
parser.add_argument('--layer', type=int, default=2, help='Number of RNN layer')
parser.add_argument('--hidden_state', type=int, default=96, help='Number of hidden state')
args = parser.parse_args()

with DBGeneticReader(args.data, read_first_k_table=args.read_first_k, two_class=True,
                     normalize=args.normalize) as reader:
    with torch.cuda.device(args.gpu):
        if args.read_old_model is None:
            model = rnn.RNNModel(64, rnn_len=args.layer, hidden_state=args.hidden_state)
            if args.cuda:
                model.cuda()
        else:
            model = torch.load(args.read_old_model)

        cc = CrayonClient(hostname=args.rec_url)
        exp = cc.create_experiment('Future_{}_{}'.format(args.name,
                                                         datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")))
        rnn.train(model, reader, exp, args, use_cuda=args.cuda)
