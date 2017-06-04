import pandas as pd
import os
import math
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Normalizer')
parser.add_argument('--dir', type=str, default='./processed', help='Iterate under the directory')
parser.add_argument('--file', type=str, default='', help='Normalize this file')
args = parser.parse_args()


def normalize_file(path):
    print('=== Normalize h5 file {}'.format(path))
    with pd.HDFStore(path, complevel=9, complib='blosc') as store:
        tables = []
        for k in store:
            print(' -- Processing table {}'.format(k))

            fill_table = {
                'AskPrice1': ['AskPrice2', 'AskPrice3', 'AskPrice4', 'AskPrice5'],
                'BidPrice1': ['BidPrice2', 'BidPrice3', 'BidPrice4', 'BidPrice5'],
                'AskVolume1': ['AskVolume2', 'AskVolume3', 'AskVolume4', 'AskVolume5'],
                'BidVolume1': ['BidVolume2', 'BidVolume3', 'BidVolume4', 'BidVolume5']
            }

            for source in fill_table:
                for target in fill_table[source]:
                    store[k][target] = np.where(store[k][target] == 0, store[k][source], store[k][target])

            t = store[k]
            t = t[t['AskVolume1'] < 100000][t['Turnover'] >= 0][t['LastTurnover'] >= 0]
            t[['Turnover', 'LastTurnover']] = t[['Turnover', 'LastTurnover']].applymap(
                lambda x: math.log(x + 1))

            tables.append(t.iloc[:, 4:].astype('float32'))

        big_table = pd.concat(tables, copy=False)
        mean = big_table.mean()
        std = big_table.std()

        print('Mean=')
        print(mean)

        print('Std=')
        print(std)
        for k in store:
            print(' -- Writing back to table {}'.format(k))
            store[k].iloc[:, 4:] = (store[k].iloc[:, 4:] - mean) / (std + 0.00001)




if args.file == '':
    for root, dirs, files in os.walk('./processed'):
        for filename in files:
            if filename.endswith('.h5'):
                fullpath = os.path.join(root, filename)
                normalize_file(fullpath)
else:
    normalize_file(args.file)
