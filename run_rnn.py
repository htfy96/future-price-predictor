from reader.genetic import DBGeneticReader
import model.rnn as rnn
from pycrayon import CrayonClient
import datetime
from torch.utils.data.sampler import RandomSampler
import argparse


parser = argparse.ArgumentParser(description='RNN Future predictor')
parser.add_argument('--data', type=str, default='./processed/m0000.h5', help='location of DB file')
parser.add_argument('--cuda', action='store_true', help='whether use cuda')
parser.add_argument('--rec_url', type=str, default='orzserver.intmainreturn0.com', help='TensorBoard address')
args = parser.parse_args()

with DBGeneticReader(args.data, read_first_k_table=1) as reader:
    model = rnn.RNNModel(64, rnn_len=5)
    if args.cuda:
        model.cuda()
    cc = CrayonClient(hostname=args.rec_url)
    exp = cc.create_experiment('DBExportRNN{}'.format(datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")))
    rnn.train(model, reader, exp, use_cuda=args.cuda)