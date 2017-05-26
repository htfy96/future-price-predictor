from reader.genetic import DBGeneticReader
import model.rnn as rnn
from pycrayon import CrayonClient
import datetime
from torch.utils.data.sampler import RandomSampler

with DBGeneticReader('./processed/DBExport.h5', read_first_k_table=5) as reader:
    model = rnn.RNNModel(64, rnn_len=5)
    cc = CrayonClient(hostname="orzserver.intmainreturn0.com")
    exp = cc.create_experiment('DBExportRNN{}'.format(datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")))
    rnn.train(model, reader, exp)