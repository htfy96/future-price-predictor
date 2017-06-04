import pandas as pd
import os
import argparse

parser = argparse.ArgumentParser(description='Preprocessor')
parser.add_argument('--source', type=str, default='./data', help='location of csv files')
parser.add_argument('--target', type=str, default='./processed', help='location of processed h5 files')
parser.add_argument('--prefix', type=str, default='v2', help='Prefix added to output name')
args = parser.parse_args()

for root, dirs, files in os.walk(args.source):
    for filename in files:
        if filename.endswith('.csv'):
            fullpath = os.path.join(root, filename)
            print('Processing data from csv={}'.format(fullpath))
            frame = pd.read_csv(fullpath)
            output_filename = os.path.join(args.target, args.prefix + '-' + os.path.split(root)[-1] + '.h5')

            chars_to_replace = ('[', ']', '.', ' ')
            str_to_replace = filename.split('.')[0]
            for c in chars_to_replace:
                str_to_replace = str_to_replace.replace(c, '_')
            output_table = str_to_replace
            print('Saving to {} with table={}'.format(output_filename, output_table))
            frame.to_hdf(output_filename, output_table, complevel=9, complib='blosc')
