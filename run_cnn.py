import reader.genetic as DBGR
import model.cnnT1 as cnn
#from pycrayon import CrayonClient
import datetime
from torch.utils.data.sampler import RandomSampler
import argparse
import torch

cuda_ava = torch.cuda.is_available()

parser = argparse.ArgumentParser(description='CNN Future predictor')
parser.add_argument('--data', type=str, default='./processed/m0000.h5', help='location of DB file')
parser.add_argument('--cuda', action='store_true', default=True, help='whether use cuda')
#parser.add_argument('--rec_url', type=str, default='orzserver.intmainreturn0.com', help='TensorBoard address')
parser.add_argument('--name', type=str, default='CNNT1', help='Name of this experiment')
parser.add_argument('--read_first_k', type=int, default=1, help='Only read first k table')
parser.add_argument('--read_old_model', type=str, default=None, help='Path of old model')
args = parser.parse_args()

with DBGR.DBGeneticReader('./processed/m0000.h5', read_first_k_table=args.read_first_k) as db:
    #dataset = []
    #for i in range(320):
    #    dataset.append(db[i])
    if args.read_old_model is None:
        model = cnn.cnnT1()
        if args.cuda and cuda_ava:
            model = model.cuda()
    else:
        model = torch.load(args.read_old_model)

    #cc = CrayonClient(hostname=args.rec_url)
    #exp = cc.create_experiment('Future_{}_{}'.format(args.name,
    #                                                 datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")))

    #cnn.train(model, dataset, args, use_cuda=args.cuda)
    cnn.train(model, db, args, eph=1, use_cuda=args.cuda)